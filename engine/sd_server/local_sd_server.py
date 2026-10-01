"""FaceSwitch 本地 SDXL 生图服务。

提供 SD WebUI (A1111) 兼容接口, FaceSwitch 的「SD WebUI」模板可直接使用:
  POST /sdapi/v1/txt2img  {prompt, width, height, steps, cfg_scale, ...}
      -> {"images": ["<png base64>"], ...}
  GET  /sdapi/v1/sd-models          -> 已加载模型列表
  GET  /sdapi/v1/options            -> 选项 (部分工具探测用)

生命周期: FaceSwitch 应用启动时由 Electron 主进程自动拉起, 退出时自动停止,
无需手动运行本脚本 (手动运行仍然可用, 但下次打开应用时会被接管重启)。
端口: 默认 127.0.0.1:7860, 可用环境变量 FACESWITCH_SD_PORT 覆盖。
内存: cpu offload 下权重常驻内存约 7GB; 空闲 15 分钟 (FACESWITCH_SD_IDLE_UNLOAD_MIN,
0 关闭) 自动卸载, 下次生成自动重载 (约 1-2 分钟)。
模型: SDXL base 1.0 fp16, 存于 HF_HOME 缓存。
"""
import base64
import gc
import io
import os
import sys
import threading
import time

# 必须在导入 torch/diffusers 前设置缓存与镜像位置
DATA_ROOT = os.environ.get("FACESWITCH_DATA_ROOT", r"D:\FaceSwitchData\SD")
os.environ.setdefault("HF_HOME", os.path.join(DATA_ROOT, "hf_cache"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 模型已在本地缓存: 默认离线加载, 不受 hf-mirror 网络抖动影响
# (如需在线拉取新模型, 设 FACESWITCH_SD_ONLINE=1)
if os.environ.get("FACESWITCH_SD_ONLINE", "") != "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

app = FastAPI(title="FaceSwitch SDXL Service", docs_url=None)

_state = {"pipe": None, "ready": False, "error": None, "placement": "", "started": time.time()}
_idle_min = float(os.environ.get("FACESWITCH_SD_IDLE_UNLOAD_MIN", "15") or 0)
_unload_lock = threading.Lock()
_unload_timer: threading.Timer | None = None


def _unload_pipeline() -> None:
    """空闲卸载: 释放权重与显存, 下次请求自动重载。"""
    with _unload_lock:
        if _state["pipe"] is None:
            return
        print("[sd] idle timeout, unloading pipeline to free memory.", flush=True)
        _state["pipe"] = None
        _state["ready"] = False
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def _schedule_unload() -> None:
    global _unload_timer
    if _idle_min <= 0:
        return
    with _unload_lock:
        if _unload_timer is not None:
            _unload_timer.cancel()
        _unload_timer = threading.Timer(_idle_min * 60, _unload_pipeline)
        _unload_timer.daemon = True
        _unload_timer.start()


@app.post("/unload")
async def unload():
    """视频任务开工前由引擎调用: 立即释放 SDXL 权重, 给换脸/抠像腾内存。"""
    _unload_pipeline()
    return {"ok": True, "loaded": False}


def _local_snapshot_dir() -> str:
    """解析 HF 缓存中 SDXL 的本地快照目录 (refs/main -> snapshots/<commit>)。"""
    from pathlib import Path

    cache = Path(os.environ.get("HF_HOME", os.path.join(DATA_ROOT, "hf_cache")))
    hub = cache / "hub" / "models--stabilityai--stable-diffusion-xl-base-1.0"
    ref = hub / "refs" / "main"
    if not ref.exists():
        return ""
    try:
        commit = ref.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    snap = hub / "snapshots" / commit
    if (snap / "model_index.json").exists():
        return str(snap)
    return ""


def _apply_placement(pipe, force_offload: bool = False) -> str:
    """显存够 (>=6.5GB 空闲) 时整管直载 GPU, 否则回退 cpu offload。

    cpu offload 权重常驻内存且每步搬运, 是"内存高 + GPU 利用率低"的主因;
    SDXL fp16 直载约 5GB, 换脸引擎的 inswapper 等模型很小, 两者可共存。
    """
    import torch

    if torch.cuda.is_available() and not force_offload:
        try:
            free_b, _total = torch.cuda.mem_get_info()
            if free_b >= int(6.5 * 1024**3):
                pipe.to("cuda")
                return "gpu"
        except Exception:  # noqa: BLE001
            pass
    pipe.enable_model_cpu_offload()
    return "offload"


def get_pipeline(force_offload: bool = False):
    """懒加载 SDXL; 默认尝试 GPU 直载, 失败/显存不足回退 cpu offload。

    加载失败不粘死: 错误只记录用于报错展示, 下次请求会重新尝试加载。
    """
    global _unload_timer
    with _unload_lock:
        if _unload_timer is not None:
            _unload_timer.cancel()
            _unload_timer = None
        if _state["pipe"] is not None:
            return _state["pipe"]
    try:
        import torch
        from diffusers import StableDiffusionXLPipeline

        # 加载前先归还本进程内存, 再触发系统级回收: 换脸引擎运行中加载 SDXL,
        # 任何残留都会叠加到加载峰值上 (DefaultCPUAllocator OOM 的直接诱因)
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

        model_dir = os.environ.get("SDXL_MODEL_DIR", "")
        if not (model_dir and os.path.isdir(model_dir)):
            # 从 HF 缓存快照目录直接加载: 绕过 Hub 层的仓库完整性检查
            # (缓存只含 diffusers 需要的文件, 快照级 completeness 校验会误报缺失)
            model_dir = _local_snapshot_dir()
        load_kwargs = dict(
            torch_dtype=torch.float16,
            variant="fp16",
            # 流式加载权重 (不整份读入内存再转 dtype), 模型加载期 CPU 内存峰值减半;
            # 16GB 机器上换脸引擎与 SDXL 共存, 加载峰值过高会触发 CPU OOM
            low_cpu_mem_usage=True,
        )
        if model_dir and os.path.isdir(model_dir):
            pipe = StableDiffusionXLPipeline.from_pretrained(model_dir, **load_kwargs)
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", **load_kwargs
            )
        placement = _apply_placement(pipe, force_offload)
        # 8GB 显存与换脸引擎共享: VAE 解码是生成全程的显存峰值, tiling/slicing
        # 分块解码削峰, 1024 分辨率下不再触发 CUDA OOM
        # 旧版 diffusers 方法挂在 vae 上, 新版提为 pipeline 方法, 两者都试
        for fn, attr in ((pipe, "enable_vae_slicing"), (pipe.vae, "enable_slicing")):
            if hasattr(fn, attr):
                getattr(fn, attr)()
                break
        for fn, attr in ((pipe, "enable_vae_tiling"), (pipe.vae, "enable_tiling")):
            if hasattr(fn, attr):
                getattr(fn, attr)()
                break
        pipe.set_progress_bar_config(disable=True)
        with _unload_lock:
            _state["pipe"] = pipe
            _state["placement"] = placement
            _state["ready"] = True
            _state["error"] = None
        print(f"[sd] SDXL loaded ({placement}), ready.", flush=True)
    except Exception as e:  # noqa: BLE001
        import traceback

        _state["error"] = f"{type(e).__name__}: {e}"
        print("[sd] load failed:", _state["error"], flush=True)
    return _state["pipe"]


SDXL_MAX_SIDE = int(os.environ.get("FACESWITCH_SD_MAX_SIDE", "1024"))


def _fit_sdxl_size(w: int, h: int) -> tuple[int, int]:
    """缩到 SDXL 原生分辨率 (长边 <=1024, 64 的倍数)。

    按视频分辨率直接生成 (1080p/4K) 会挤爆 8GB 显存; FaceSwitch 拿到背景图
    后还会按视频尺寸重新缩放, 所以生成小图不影响最终质量。
    """
    scale = min(1.0, SDXL_MAX_SIDE / max(w, h))
    w2 = max(round(w * scale / 64) * 64, 256)
    h2 = max(round(h * scale / 64) * 64, 256)
    return w2, h2


class Txt2ImgReq(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = Field(default=1024, ge=256, le=4096)
    height: int = Field(default=1024, ge=256, le=4096)
    steps: int = Field(default=25, ge=1, le=60)
    cfg_scale: float = Field(default=6.5, ge=1.0, le=20.0)
    seed: int | None = None


@app.post("/sdapi/v1/txt2img")
async def txt2img(req: Txt2ImgReq):
    pipe = get_pipeline()
    if pipe is None:
        return JSONResponse(status_code=503, content={"error": _state["error"] or "模型加载中"})
    try:
        import torch

        generator = None
        if req.seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(req.seed)
        w, h = _fit_sdxl_size(req.width, req.height)
        kwargs = dict(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt or None,
            width=w,
            height=h,
            num_inference_steps=req.steps,
            guidance_scale=req.cfg_scale,
            generator=generator,
        )
        try:
            result = pipe(**kwargs)
        except Exception as e:  # noqa: BLE001 — OOM: 清缓存重试, 再失败降级 offload 重试
            if "out of memory" not in str(e).lower():
                raise
            print("[sd] CUDA OOM, retry after empty_cache:", e, flush=True)
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            try:
                result = pipe(**kwargs)
            except Exception as e2:  # noqa: BLE001
                if "out of memory" not in str(e2).lower() or _state.get("placement") != "gpu":
                    raise
                # GPU 直载下仍 OOM: 整管降级到 cpu offload 再试一次
                print("[sd] still OOM on gpu placement, falling back to offload.", flush=True)
                pipe.to("cpu")
                torch.cuda.empty_cache()
                pipe.enable_model_cpu_offload()
                _state["placement"] = "offload"
                result = pipe(**kwargs)
        img = result.images[0]
        buf = io.BytesIO()
        img.save(buf, format="PNG")
    except Exception as e:  # noqa: BLE001 — 显存不足/生成异常, 返回可读错误而非裸 500
        import traceback

        detail = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-400:]}"
        print("[sd] txt2img failed:", detail, flush=True)
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        return JSONResponse(status_code=503, content={"error": detail[:600]})
    _schedule_unload()
    return {"images": [base64.b64encode(buf.getvalue()).decode()], "info": "{}"}


@app.get("/sdapi/v1/sd-models")
async def models():
    return [{"title": "sdxl-base-1.0-fp16", "model_name": "sdxl-base-1.0-fp16"}]


@app.get("/sdapi/v1/options")
async def options():
    return {"sd_model_checkpoint": "sdxl-base-1.0-fp16"}


@app.get("/health")
async def health():
    return {
        "service": "faceswitch-sdxl",  # 标识: Electron 用它区分本服务与端口上的其他服务
        "ready": _state["ready"],
        "error": _state["error"],
        "uptimeSec": round(time.time() - _state["started"]),
    }


SD_PORT = int(os.environ.get("FACESWITCH_SD_PORT", "7860"))

if __name__ == "__main__":
    print(f"[sd] starting on 127.0.0.1:{SD_PORT} ...", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=SD_PORT, log_level="warning")
