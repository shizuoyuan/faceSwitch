# FaceSwitch — 视频换脸换背景专业桌面软件 实现方案

## 目标
Windows 桌面应用:上传视频 → 换脸 / 换背景 → 导出。本地 GPU(RTX 5060 Ti 8GB)跑换脸与抠像,AI 生成背景走**用户可配置的远程 API**(你的本地 AI / 万相 / Replicate 均可)。

## 环境现状(已探明)
- GPU: RTX 5060 Ti 8GB,Blackwell 架构(sm_120)→ **必须 torch≥2.7 cu128、onnxruntime-gpu≥1.20**
- 未安装: Node.js、Python(仅商店占位符)、ffmpeg → 需引导安装
- 中国网络 → 全链路走镜像(npmmirror / 清华 pip / hf-mirror.com)
- 空目录,非 git 仓库 → 开工先 `git init`

## 技术栈
| 层 | 选型 | 理由 |
|---|---|---|
| 桌面壳 | Electron + Vite + React + TS + Zustand | 生态成熟、迭代快;中文向导式 UI、暗色专业风 |
| 引擎 | Python 3.11 + FastAPI(localhost sidecar,HTTP+SSE 进度) | Electron 主进程拉起/守护/退出清理 |
| 换脸 | **纯 onnxruntime 栈**(FaceFusion 路线):SCRFD 检测 + ArcFace 识别 + inswapper_128 换脸 + GFPGAN ONNX 修复 | 避开 insightface 包在 Windows 的编译坑;8GB 显存充裕 |
| 抠像 | RVM (RobustVideoMatting, PyTorch, mobilenetv3) | 时序天然稳定;权重 CC-BY-NC(自用无碍,商用再换 BiRefNet/MIT) |
| 融合 | Reinhard 色彩迁移(前景匹配背景光照,参数时序 EMA 平滑) | 轻量"重打光"首版;IC-Light 留作升级位 |
| 时序 | 换脸区域 EMA + 检测框平滑(opencv Farneback 光流备选) | 抗闪烁,独立成环节 |
| 视频 | ffmpeg(imageio-ffmpeg 内置二进制,可被系统 ffmpeg 覆盖) | 抽帧/合帧/原音轨 copy/libx264 CRF18 |

## 处理流水线(每环节=插件,实现 Stage 协议,可单独替换)
```
①抽帧(ffprobe/ffmpeg) → ②RVM 抠像 → ③背景
   [保留原背景 | 上传图片 | 上传视频(循环/裁剪) | AI生成(调配置的远程API)]
→ ④色彩匹配融合 → ⑤换脸(检测→源脸ArcFace匹配→inswapper→GFPGAN,支持多脸)
→ ⑥时序平滑 → ⑦合成导出(mp4+原音轨)
```

## 远程 AI 层(核心需求:你的本地 AI 通过 API 地址配置)
设置页管理多个 Provider,每个含:名称、Base URL、Header(含Key)、请求体 JSON 模板(占位符 `{prompt}` `{width}` `{height}` `{first_frame_b64}`)、响应解析路径(如 `data.output_url`,支持返回 URL 或 base64)。内置万相/Replicate 预设模板,自定义类型对接你的本地 AI。不配置任何 Provider 时软件仍完整可用(本地背景路线)。

## 界面(中文向导 4 步 + 设置)
1. **导入**:拖拽视频,显示时长/分辨率/缩略图(首版建议 ≤1080p、≤2 分钟)
2. **配置**:换脸开关+源脸图(多张);背景四种模式;高级项(修复强度、色匹配、输出质量)
3. **处理**:分阶段进度、实时帧预览、可取消
4. **导出**:内嵌播放、打开文件夹、再来一单
设置页:Provider 管理、模型目录、GPU/CPU 选择

## 目录结构
```
face-switch/
├─ electron/{main.ts, preload.ts}      # 窗口、sidecar 守护、IPC
├─ src/pages/{Import,Configure,Process,Export,Settings}.tsx
├─ engine/
│  ├─ api.py                            # FastAPI /jobs /settings /models
│  ├─ pipeline/{extract,matting,background,harmonize,faceswap,temporal,compose}.py
│  ├─ models/{scrfd,arcface,swapper,restorer,rvm}.py   # onnx/torch 推理封装
│  ├─ providers/remote.py               # 模板引擎+预设
│  ├─ downloads.py                      # 权重下载(hf-mirror,首启下载)
│  └─ requirements.txt
├─ scripts/bootstrap.cmd                # 工具链+镜像引导
└─ models/                              # 权重缓存(gitignore)
```

## 里程碑
- **M0 工具链引导**:装 Node LTS + Python 3.11,配镜像,建 venv,验证 torch cu128 与 onnxruntime-gpu 在 5060 Ti 上真的能用
- **M1 骨架**:Electron 壳 + sidecar 生命周期 + 设置持久化 + 导入页
- **M2 换背景(本地路线)**:抽帧→RVM→本地背景合成→色匹配→编码导出,打通全流程
- **M3 换脸**:全套 onnx 推理 + 时序平滑 + 多脸
- **M4 远程 AI 层**:Provider 配置 UI + 模板引擎 + AI 生成背景
- **M5 打磨**:暂停恢复、任务队列、electron-builder + PyInstaller 打包安装程序

## 风险与对策
- **Blackwell 兼容**:锁 torch≥2.7+cu128;onnxruntime-gpu 失败自动回退 CPU(检测/换脸 CPU 可跑,RVM 必须 GPU)
- **时序一致性是最大质量风险**:已独立成环节,RVM 自带时序、换脸区 EMA 平滑,首版接受折中
- **权重许可**:inswapper/GFPGAN 研究用途、RVM 非商用——自用没问题,商用需替换,代码里模块化预留
- **性能预期**:1080p30 全流水线约 8~15 fps 处理速度,1 分钟视频约 5~8 分钟出片

## 实施所需权限
- 安装开发工具链(Node.js、Python、构建工具)
- npm/pip 安装依赖(走国内镜像)
- 下载 AI 模型权重文件(hf-mirror)
- git init 与提交
- 启动/调试 Electron 与 Python 引擎、运行 ffmpeg 处理视频