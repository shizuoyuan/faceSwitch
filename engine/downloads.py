"""模型权重下载管理(全部走 hf-mirror,URL 已验证)。"""
import threading
from pathlib import Path

import httpx

from .config import hf_mirror, writable_models_dir

FACEFUSION = f"{hf_mirror()}/facefusion/models-3.0.0/resolve/main"
RVM_REPO = f"{hf_mirror()}/inverseaibd/rvm_mobilenetv3_fp16/resolve/main"

MODEL_MANIFEST: dict[str, str] = {
    "scrfd_2.5g.onnx": f"{FACEFUSION}/scrfd_2.5g.onnx",
    "arcface_w600k_r50.onnx": f"{FACEFUSION}/arcface_w600k_r50.onnx",
    "inswapper_128.onnx": f"{FACEFUSION}/inswapper_128.onnx",
    "gfpgan_1.4.onnx": f"{FACEFUSION}/gfpgan_1.4.onnx",
    "rvm_mobilenetv3_fp16.onnx": f"{RVM_REPO}/rvm_mobilenetv3_fp16.onnx",
}

_progress: dict[str, dict] = {}
_lock = threading.Lock()


def target_dir() -> str:
    return str(writable_models_dir())


def status(models_dir: str) -> dict[str, bool]:
    d = Path(models_dir)
    return {name: (d / name).exists() for name in MODEL_MANIFEST}


def progress() -> dict:
    with _lock:
        return {k: dict(v) for k, v in _progress.items()}


def download_all(models_dir: str) -> None:
    """后台线程里下载全部缺失权重;进度记录在 _progress。"""
    threading.Thread(
        target=_download_worker, args=(models_dir,), daemon=True
    ).start()


def _download_worker(models_dir: str) -> None:
    d = Path(models_dir)
    d.mkdir(parents=True, exist_ok=True)
    for name, url in MODEL_MANIFEST.items():
        dest = d / name
        if dest.exists():
            continue
        with _lock:
            _progress[name] = {"done": False, "percent": 0, "error": ""}
        try:
            with httpx.stream("GET", url, timeout=60, follow_redirects=True) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                got = 0
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=1 << 20):
                        f.write(chunk)
                        got += len(chunk)
                        if total:
                            with _lock:
                                _progress[name]["percent"] = int(got * 100 / total)
                tmp.rename(dest)
                with _lock:
                    _progress[name]["done"] = True
                    _progress[name]["percent"] = 100
        except Exception as e:  # noqa: BLE001
            with _lock:
                _progress[name]["error"] = str(e)
