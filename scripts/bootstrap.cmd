@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0.."

echo ============================================
echo   FaceSwitch 一键引导 (Node + Python + 依赖)
echo ============================================

where winget >nul 2>&1
if errorlevel 1 (
  echo [错误] 未找到 winget, 请手动安装 Node.js LTS 与 Python 3.11
  pause & exit /b 1
)

where python >nul 2>&1
if errorlevel 1 (
  echo [1/5] 安装 Python 3.11 ...
  winget install -e --id Python.Python.3.11 --scope user --accept-package-agreements --accept-source-agreements
) else (
  echo [1/5] Python 已安装
)

where node >nul 2>&1
if errorlevel 1 (
  echo [2/5] 安装 Node.js LTS ...
  winget install -e --id OpenJS.NodeJS.LTS --accept-package-agreements --accept-source-agreements
) else (
  echo [2/5] Node.js 已安装
)

set "PY=%CD%\engine\.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [3/5] 创建 Python 虚拟环境 ...
  python -m venv engine\.venv
)

echo [4/5] 安装 Python 依赖 (PyTorch CUDA 12.8 约 2.7GB, 请耐心等待) ...
"%PY%" -m pip install -q -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
"%PY%" -m pip install torch --index-url https://download.pytorch.org/whl/cu128
"%PY%" -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r engine\requirements.txt "onnxruntime-gpu==1.23.2"

echo [5/5] 安装 Node 依赖 ...
call npm install

echo.
echo 引导完成! 启动应用: npm run dev
pause
