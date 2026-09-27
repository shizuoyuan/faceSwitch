# -*- mode: python ; coding: utf-8 -*-
"""engine.spec — PyInstaller onedir 打包 FaceSwitch Python 引擎。

产出 engine/dist/engine/ (engine.exe + _internal/)。
CUDA DLL 从 torch/lib 收集到 _internal/dlls, 运行时由 engine.models.base._add_dll_dirs 注入。
"""
import glob
import os
from PyInstaller.utils.hooks import collect_submodules

torch_lib = os.path.join("engine", ".venv", "Lib", "site-packages", "torch", "lib")
cuda_dlls = []
for pat in ("cudart64*.dll", "cublas*.dll", "cudnn*.dll"):
    cuda_dlls += sorted(glob.glob(os.path.join(torch_lib, pat)))

a = Analysis(
    ["run_engine.py"],
    pathex=["."],
    binaries=[(d, "dlls") for d in cuda_dlls],
    datas=[],
    hiddenimports=collect_submodules("uvicorn") + ["engine.api"],
    excludes=[
        "torch",
        "torchvision",
        "torchaudio",
        "matplotlib",
        "tkinter",
        "IPython",
        "jupyter",
        "pytest",
        "PyQt5",
        "PySide2",
        "PySide6",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="engine",
    debug=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="engine",
)
