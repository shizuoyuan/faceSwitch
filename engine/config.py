"""路径与用户设置管理。"""
import json
import os
from pathlib import Path
from typing import Any

ENGINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ENGINE_DIR.parent
DATA_DIR = ENGINE_DIR / "data"
SETTINGS_FILE = DATA_DIR / "settings.json"

DEFAULT_SETTINGS: dict[str, Any] = {
    "providers": [],
    "device": "gpu",
    "modelsDir": str(PROJECT_ROOT / "models"),
    "outputDir": str(PROJECT_ROOT / "output"),
}


def load_settings() -> dict[str, Any]:
    if SETTINGS_FILE.exists():
        try:
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            merged = {**DEFAULT_SETTINGS, **data}
            return merged
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


def hf_mirror() -> str:
    return os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
