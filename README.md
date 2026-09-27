# FaceSwitch — AI 视频换脸 · 换背景 桌面工作台

上传视频,在本地 GPU 上完成 **换脸** 与 **换背景**,专业流水线架构,人脸数据不出本机。

## 功能

- **AI 换脸**:SCRFD 检测 + ArcFace 识别 + inswapper 换脸 + GFPGAN 修复,支持多源脸智能匹配、关键点时序平滑抗闪烁
- **换背景**(四种模式):保留原背景 / 上传图片 / 上传视频循环 / **AI 生成**(对接你自己的 AI 服务)
- **专业流水线**:RVM 时序抠像 → 背景合成 → Reinhard 色彩匹配(光照融合)→ 换脸 → 编码,单遍流式处理
- **远程 AI 层**:设置页可配置任意 HTTP 图像生成服务(SD WebUI / OpenAI 兼容接口 / 自定义),填地址+模板即可接入

## 环境要求

- Windows 10/11 + NVIDIA 显卡(8GB 显存测试通过,RTX 50 系需 CUDA 12.8 版 PyTorch)
- 首次使用运行 `scripts\bootstrap.cmd` 一键引导(Node.js / Python 3.11 / 依赖)
- 首次处理任务时自动从 hf-mirror 下载模型权重(约 1.1GB,仅一次)

## 启动

```bash
npm run dev
```

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
