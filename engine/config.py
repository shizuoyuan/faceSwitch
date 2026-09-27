"""路径与用户设置管理。

开发模式: 数据与模型都在项目目录。
打包模式 (PyInstaller + electron-builder extraResources):
  - 引擎与 DLL 在 resources/engine/, 模型在 resources/models/ (只读)
  - 设置/任务/输出/补下载的模型写到 %APPDATA%/FaceSwitch/
"""
import json
import os
import sys
from pathlib import Path
from typing import Any

ENGINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ENGINE_DIR.parent

IS_FROZEN = bool(getattr(sys, "frozen", False))

if IS_FROZEN:
    # onedir: sys._MEIPASS = <resources>/engine/_internal
    _bundle_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    BUNDLED_MODELS_DIR = _bundle_root.parents[1] / "models"  # resources/models
    _app_data = Path(os.environ.get("APPDATA") or Path.home()) / "FaceSwitch"
    DATA_DIR = _app_data
    FALLBACK_MODELS_DIR = _app_data / "models"
    DEFAULT_OUTPUT_DIR = _app_data / "output"
else:
    BUNDLED_MODELS_DIR = PROJECT_ROOT / "models"
    DATA_DIR = ENGINE_DIR / "data"
    FALLBACK_MODELS_DIR = BUNDLED_MODELS_DIR
    DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"

SETTINGS_FILE = DATA_DIR / "settings.json"

DEFAULT_SETTINGS: dict[str, Any] = {
    "providers": [],
    "device": "gpu",
    "modelsDir": str(BUNDLED_MODELS_DIR),
    "outputDir": str(DEFAULT_OUTPUT_DIR),
}


def load_settings() -> dict[str, Any]:
    if SETTINGS_FILE.exists():
        try:
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            return {**DEFAULT_SETTINGS, **data}
        except (json.JSONDecodeError, OSError):
            pass
    return dict(DEFAULT_SETTINGS)


def save_settings(new: dict[str, Any]) -> dict[str, Any]:
    merged = {**DEFAULT_SETTINGS, **new}
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return merged


def resolve_model(filename: str, settings: dict[str, Any] | None = None) -> Path:
    """按优先级查找模型: 设置目录 -> APPDATA 补下载目录 -> 安装包内置目录。"""
    s = settings or load_settings()
    for base in (s.get("modelsDir"), FALLBACK_MODELS_DIR, BUNDLED_MODELS_DIR):
        if base:
            p = Path(base) / filename
            if p.exists():
                return p
    raise FileNotFoundError(f"模型缺失且未下载: {filename}")


def writable_models_dir(settings: dict[str, Any] | None = None) -> Path:
    """下载目标: 设置目录不可写时 (如装进 Program Files) 回退 APPDATA。"""
    s = settings or load_settings()
    d = Path(s.get("modelsDir") or FALLBACK_MODELS_DIR)
    try:
        d.mkdir(parents=True, exist_ok=True)
        probe = d / ".writetest"
        probe.touch()
        probe.unlink()
        return d
    except OSError:
        FALLBACK_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        return FALLBACK_MODELS_DIR


def hf_mirror() -> str:
    return os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
