"""ONNX 会话管理与通用工具。"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def _add_dll_dirs() -> None:
    """Windows 上让 onnxruntime 找到 cudnn/cublas DLL。

    开发模式借 torch/lib; 打包模式从 _internal/dlls 取。
    ort 加载 provider 依赖时走 PATH 搜索, 因此同时改 add_dll_directory 与 PATH。
    """
    import site

    dirs: list[str] = []
    if getattr(sys, "frozen", False):
        meipass = Path(sys._MEIPASS)  # type: ignore[attr-defined]
        for sub in ("dlls", "onnxruntime/capi"):
            p = meipass / sub
            if p.exists():
                dirs.append(str(p))
    else:
        for sp in site.getsitepackages():
            torch_lib = Path(sp) / "torch" / "lib"
            if torch_lib.exists():
                dirs.append(str(torch_lib))
            nvidia = Path(sp) / "nvidia"
            if nvidia.exists():
                for sub in nvidia.iterdir():
                    bin_dir = sub / "bin"
                    if bin_dir.exists():
                        dirs.append(str(bin_dir))
    for d in dirs:
        try:
            os.add_dll_directory(d)
        except OSError:
            pass
    if dirs:
        os.environ["PATH"] = os.pathsep.join(dirs) + os.pathsep + os.environ.get("PATH", "")


def create_session(model_path: str, device: str = "gpu") -> "ort.InferenceSession":
    import onnxruntime as ort

    model_path = str(model_path)
    if not Path(model_path).exists():
        raise FileNotFoundError(f"模型缺失: {model_path}")

    wanted = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "gpu"
        else ["CPUExecutionProvider"]
    )
    available = set(ort.get_available_providers())
    providers = [p for p in wanted if p in available]
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3
    session = ort.InferenceSession(
        model_path, sess_options=sess_opts, providers=providers or ["CPUExecutionProvider"]
    )
    return session


def session_device(session: "ort.InferenceSession") -> str:
    return (
        "gpu"
        if "CUDAExecutionProvider" in session.get_providers()
        else "cpu"
    )


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.4) -> list[int]:
    """标准 NMS,返回保留索引。"""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def resize_letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float]:
    """保持宽高比缩放并右/下补齐到 size x size,返回 (图, 缩放系数)。"""
    h, w = img.shape[:2]
    im_ratio = h / w
    if im_ratio > 1:
        new_h, new_w = size, int(round(size / im_ratio))
    else:
        new_w, new_h = size, int(round(size * im_ratio))
    scale = new_h / h if im_ratio > 1 else new_w / w
    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.zeros((size, size, 3), dtype=img.dtype)
    canvas[:new_h, :new_w] = resized
    return canvas, scale


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Umeyama 相似变换,返回 2x3 仿射矩阵 (s*R|t)。"""
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / len(src)
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[1, 1] = -1
    r = u @ s @ vt
    var_s = (sc**2).sum() / len(src)
    c = (d * np.diag(s)).sum() / max(var_s, 1e-9)
    t = mu_d - c * r @ mu_s
    m = np.zeros((2, 3))
    m[:, :2] = c * r
    m[:, 2] = t
    return m
