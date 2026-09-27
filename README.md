# FaceSwitch — AI 视频换脸 · 换背景 桌面工作台

上传视频,在本地 GPU 上完成 **换脸** 与 **换背景**,专业流水线架构,人脸数据不出本机。

## 功能

- **AI 换脸**:SCRFD 检测 + ArcFace 识别 + inswapper 换脸 + GFPGAN 修复,支持多源脸智能匹配、关键点时序平滑抗闪烁
- **换背景**(四种模式):保留原背景 / 上传图片 / 上传视频循环 / **AI 生成**(对接你自己的 AI 服务)
- **专业流水线**:RVM 时序抠像 → 背景合成 → Reinhard 色彩匹配(光照融合)→ 换脸 → 编码,单遍流式处理
- **远程 AI 层**:设置页可配置任意 HTTP 图像生成服务(SD WebUI / OpenAI 兼容接口 / 自定义),填地址+模板即可接入
- **本地 SDXL 生图服务**:应用启动自动拉起(`127.0.0.1:7860`,A1111 兼容),退出自动停止;由机器上的 `engine\.venv`(含 torch+diffusers)运行,可用 `FACESWITCH_SD_PYTHON` 指定其他 Python 环境、`FACESWITCH_SD_PORT` 改端口

## 环境要求

- Windows 10/11 + NVIDIA 显卡(8GB 显存测试通过,RTX 50 系需 CUDA 12.8 版 PyTorch)
- 首次使用运行 `scripts\bootstrap.cmd` 一键引导(Node.js / Python 3.11 / 依赖)
- 首次处理任务时自动从 hf-mirror 下载模型权重(约 1.1GB,仅一次)

## 启动

```bash
npm run dev
```

## 打包发布

```bash
npm run build:engine   # 增量 PyInstaller 打包 Python 引擎 (源码未变化时秒级跳过)
npm run pack:dir       # 快速验证: 只产出 release/win-unpacked, 不打 NSIS 安装包
npm run package        # 完整发布: build:engine → electron-vite build → NSIS 安装包
```

说明:

- 引擎产出在 `engine/dist/engine/`(engine.exe + `_internal/`,约 2.5GB,含 CUDA/cuDNN DLL);`_internal/dlls` 必须包含 cudart/cublas/cudnn/**cufft**,缺一个 onnxruntime CUDA provider 都会加载失败并静默回退 CPU。
- 若引擎源码与 `engine.spec` 自上次构建后未变,`build:engine` 会直接复用 `engine/dist/engine`。
- **首次构建较慢**(PyInstaller 对 5GB venv 做全量分析,可能 1 小时以上),之后 Analysis 有缓存。若文件操作异常慢(复制 ~1MB/s),把项目目录和 `engine\.venv` 加入杀毒软件(Windows Defender / 360 等)的扫描排除列表。

## 模型清单(自动下载)

| 模型 | 用途 |
|---|---|
| scrfd_2.5g.onnx | 人脸检测 |
| arcface_w600k_r50.onnx | 人脸识别/源脸嵌入 |
| inswapper_128.onnx | 换脸 |
| gfpgan_1.4.onnx | 面部修复 |
| rvm_mobilenetv3_fp16.onnx | 视频人像抠像 |

## 许可说明

inswapper / GFPGAN / RVM 权重仅限研究与非商用用途;商用需替换相应模型(流水线各环节均为可插拔模块,见 `engine/pipeline.py`)。

## 架构

```
electron/        Electron 主进程 + preload (sidecar 守护)
src/             React 渲染层 (导入→配置→处理→导出 向导)
engine/          Python FastAPI 引擎
  ├─ models/     SCRFD / ArcFace / inswapper / GFPGAN / RVM 推理封装
  ├─ pipeline.py 单遍流式流水线 (抠像→背景→色匹配→换脸)
  ├─ providers/  可配置远程 AI 服务 (模板引擎)
  └─ api.py      localhost HTTP API (任务/设置/模型管理)
```
