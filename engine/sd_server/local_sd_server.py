"""FaceSwitch 本地 SDXL 生图服务。

提供 SD WebUI (A1111) 兼容接口, FaceSwitch 的「SD WebUI」模板可直接使用:
  POST /sdapi/v1/txt2img  {prompt, width, height, steps, cfg_scale, ...}
      -> {"images": ["<png base64>"], ...}
  GET  /sdapi/v1/sd-models          -> 已加载模型列表
  GET  /sdapi/v1/options            -> 选项 (部分工具探测用)

生命周期: FaceSwitch 应用启动时由 Electron 主进程自动拉起, 退出时自动停止,
无需手动运行本脚本 (手动运行仍然可用, 但下次打开应用时会被接管重启)。
端口: 默认 127.0.0.1:7860, 可用环境变量 FACESWITCH_SD_PORT 覆盖。
模型: SDXL base 1.0 fp16, 存于 HF_HOME 缓存。
"""
import base64
import io
import os
import sys
import time

# 必须在导入 torch/diffusers 前设置缓存与镜像位置
DATA_ROOT = os.environ.get("FACESWITCH_DATA_ROOT", r"D:\FaceSwitchData\SD")
os.environ.setdefault("HF_HOME", os.path.join(DATA_ROOT, "hf_cache"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

app = FastAPI(title="FaceSwitch SDXL Service", docs_url=None)

_state = {"pipe": None, "ready": False, "error": None, "started": time.time()}


def get_pipeline():
    """懒加载 SDXL; 8GB 显存用 cpu offload 保稳定。"""
    if _state["pipe"] is not None or _state["error"]:
        return _state["pipe"]
    try:
        import torch
        from diffusers import StableDiffusionXLPipeline

        model_dir = os.environ.get("SDXL_MODEL_DIR", "")
        if model_dir and os.path.isdir(model_dir):
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_dir, torch_dtype=torch.float16, variant="fp16"
            )
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16,
                variant="fp16",
            )
        pipe.enable_model_cpu_offload()  # 权重驻留内存, 按需进 GPU
        pipe.set_progress_bar_config(disable=True)
        _state["pipe"] = pipe
        _state["ready"] = True
        print("[sd] SDXL loaded, ready.", flush=True)
    except Exception as e:  # noqa: BLE001
        import traceback

        _state["error"] = f"{e}\n{traceback.format_exc()[-500:]}"
        print("[sd] load failed:", _state["error"], flush=True)
    return _state["pipe"]


class Txt2ImgReq(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)
    steps: int = Field(default=25, ge=1, le=60)
    cfg_scale: float = Field(default=6.5, ge=1.0, le=20.0)
    seed: int | None = None


@app.post("/sdapi/v1/txt2img")
async def txt2img(req: Txt2ImgReq):
    pipe = get_pipeline()
    if pipe is None:
        return JSONResponse(status_code=503, content={"error": _state["error"] or "模型加载中"})
    import torch

    generator = None
    if req.seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(req.seed)
    # 宽高取 8 的倍数, 避免潜在对齐问题
    w, h = req.width // 8 * 8, req.height // 8 * 8
    result = pipe(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt or None,
        width=w,
        height=h,
        num_inference_steps=req.steps,
        guidance_scale=req.cfg_scale,
        generator=generator,
    )
    img = result.images[0]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {"images": [base64.b64encode(buf.getvalue()).decode()], "info": "{}"}


@app.get("/sdapi/v1/sd-models")
async def models():
    return [{"title": "sdxl-base-1.0-fp16", "model_name": "sdxl-base-1.0-fp16"}]


@app.get("/sdapi/v1/options")
async def options():
    return {"sd_model_checkpoint": "sdxl-base-1.0-fp16"}


@app.get("/health")
async def health():
    return {
        "service": "faceswitch-sdxl",  # 标识: Electron 用它区分本服务与端口上的其他服务
        "ready": _state["ready"],
        "error": _state["error"],
        "uptimeSec": round(time.time() - _state["started"]),
    }


SD_PORT = int(os.environ.get("FACESWITCH_SD_PORT", "7860"))

if __name__ == "__main__":
    print(f"[sd] starting on 127.0.0.1:{SD_PORT} ...", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=SD_PORT, log_level="warning")
