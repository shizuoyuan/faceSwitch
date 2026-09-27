"""增量 PyInstaller 构建脚本 — 引擎源码与 spec 未变化且已有产物时直接跳过。

用法 (npm run build:engine):
    engine\\.venv\\Scripts\\python.exe scripts\\build_engine.py

指纹覆盖 engine/**/*.py + run_engine.py + engine.spec; 指纹与产物匹配则跳过,
否则调用当前 venv 的 PyInstaller 全量重建并写入新指纹。
"""
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "engine" / "dist" / "engine"
WORKPATH = ROOT / "engine" / "build"
STATE = WORKPATH / ".engine-src.sha256"
LOG = ROOT / "pyinstaller.log"


def fingerprint() -> str:
    files = [ROOT / "engine.spec", ROOT / "run_engine.py"]
    files += sorted((ROOT / "engine").rglob("*.py"))
    h = hashlib.sha256()
    for f in files:
        h.update(str(f.relative_to(ROOT)).replace("\\", "/").encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def up_to_date(fp: str) -> bool:
    exe = DIST / "engine.exe"
    return (
        STATE.exists()
        and STATE.read_text(encoding="utf-8").strip() == fp
        and exe.exists()
        and (DIST / "_internal" / "dlls").is_dir()
    )


def log_tail(n: int = 25) -> str:
    if not LOG.exists():
        return ""
    lines = LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    fp = fingerprint()
    if up_to_date(fp):
        print(f"[build_engine] 引擎源码未变化, 跳过 PyInstaller ({DIST})")
        return 0

    print("[build_engine] 引擎源码或 spec 有变化, 运行 PyInstaller ...")
    cmd = [
        sys.executable, "-m", "PyInstaller", "engine.spec",
        "--noconfirm",
        "--distpath", "engine/dist",
        "--workpath", "engine/build",
    ]
    with LOG.open("w", encoding="utf-8", errors="replace") as lf:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT)

    if proc.returncode != 0 or not (DIST / "engine.exe").exists():
        print(f"[build_engine] PyInstaller 失败 (exit={proc.returncode}), 日志尾部:")
        print(log_tail())
        print(f"[build_engine] 完整日志: {LOG}")
        return proc.returncode or 1

    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(fp, encoding="utf-8")
    print(f"[build_engine] 构建完成: {DIST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
