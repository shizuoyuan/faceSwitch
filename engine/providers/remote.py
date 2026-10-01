"""可配置远程 AI 服务:模板渲染 -> HTTP 调用 -> 取图。

占位符: {prompt} {width} {height} {first_frame_b64}
响应解析: responsePath 如 "images.0" / "data.0.b64_json",
responseType: base64(data URI 或裸 base64) 或 url(再下载)。
"""
import base64
import json
import re
import time

import cv2
import httpx
import numpy as np


class ProviderError(Exception):
    pass


def render(template: str, values: dict[str, str]) -> str:
    out = template
    for k, v in values.items():
        out = out.replace("{" + k + "}", v)
    return out


def _dig(obj, path: str):
    cur = obj
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict):
            if part not in cur:
                raise ProviderError(f"响应路径不存在: {path} (在 {list(cur.keys())[:8]} 中)")
            cur = cur[part]
        else:
            raise ProviderError(f"响应路径中途遇到非容器: {path}")
    return cur


def call_provider(provider: dict, prompt: str, width: int, height: int,
                  first_frame_b64: str = "") -> np.ndarray:
    """调用配置的服务,返回 BGR 图像。"""
    values = {
        "prompt": prompt,
        "width": str(width),
        "height": str(height),
        "first_frame_b64": first_frame_b64,
    }
    try:
        headers = json.loads(provider.get("headersJson") or "{}")
    except json.JSONDecodeError as e:
        raise ProviderError(f"请求头不是合法 JSON: {e}")

    body_str = render(provider.get("bodyTemplate") or "{}", values)
    try:
        body = json.loads(body_str)
    except json.JSONDecodeError as e:
        raise ProviderError(f"渲染后的请求体不是合法 JSON: {e} ({body_str[:200]})")

    url = provider.get("baseUrl", "").strip()
    if not url:
        raise ProviderError("服务地址为空")

    method = (provider.get("method") or "POST").upper()
    timeout = httpx.Timeout(300.0, connect=15.0)

    def send() -> httpx.Response:
        if method == "GET":
            return httpx.get(url, headers=headers, params=body, timeout=timeout)
        return httpx.post(url, headers=headers, json=body, timeout=timeout)

    # 连接被拒: 本地生图服务可能正被应用拉起, 短暂重试 (~12s) 再放弃
    resp: httpx.Response | None = None
    last_connect_err: Exception | None = None
    for _ in range(4):
        try:
            resp = send()
            break
        except httpx.ConnectError as e:
            last_connect_err = e
            time.sleep(3)
    if resp is None:
        raise ProviderError(
            f"无法连接 AI 服务 {url}: {last_connect_err}。"
            "若使用本地 SDXL 生图服务, 它由应用启动时自动拉起 (约需几秒), 请稍候重试;"
            "若持续失败请查看应用日志 sd_server.log"
        ) from last_connect_err

    if resp.status_code >= 400:
        raise ProviderError(f"服务返回 {resp.status_code}: {resp.text[:300]}")

    try:
        data = resp.json()
    except json.JSONDecodeError:
        raise ProviderError(f"响应不是 JSON: {resp.text[:200]}")

    value = _dig(data, provider.get("responsePath") or "")
    if not isinstance(value, str) or not value:
        raise ProviderError(f"响应取值 {provider.get('responsePath')} 不是字符串")

    if provider.get("responseType", "base64") == "url":
        try:
            img_resp = httpx.get(value, timeout=120, follow_redirects=True)
            img_resp.raise_for_status()
            payload = img_resp.content
        except httpx.HTTPError as e:
            raise ProviderError(f"下载生成图片失败: {e}")
    else:
        b64 = re.sub(r"^data:[^;]+;base64,", "", value.strip())
        try:
            payload = base64.b64decode(b64)
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"base64 解码失败: {e}")

    img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ProviderError("返回内容无法解码为图片")
    return img
