"""FaceSwitch 本地引擎 API (FastAPI sidecar)。

由 Electron 主进程拉起: python -m engine.api
鉴权: 环境变量 ENGINE_TOKEN, 请求头 X-Engine-Token 或 ?token=
"""
import base64
import os
import uuid
from pathlib import Path

import cv2
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from . import downloads
from .config import DATA_DIR, load_settings, resolve_model, save_settings
from .media import probe
from .pipeline import Job

app = FastAPI(title="FaceSwitch Engine", docs_url=None, redoc_url=None)

JOBS_ROOT = DATA_DIR / "jobs"
JOBS_ROOT.mkdir(parents=True, exist_ok=True)

_jobs: dict[str, Job] = {}
_engine_token = os.environ.get("ENGINE_TOKEN", "")

def _nudge_sd_unload() -> None:
    """开工前让本地 SDXL 服务卸载权重 (8GB 显存上两者共存的内存竞争兜底)。

    同步等待卸载完成 (SD 空闲卸载含 gc + empty_cache, 通常 <1s);
    服务未起/非本服务占用端口时静默忽略, 不影响任务。
    """
    port = os.environ.get("FACESWITCH_SD_PORT", "7860")
    try:
        import httpx

        httpx.post(f"http://127.0.0.1:{port}/unload", timeout=10.0)
    except Exception:  # noqa: BLE001
        pass

async def auth(request: Request) -> None:
    token = request.query_params.get("token") or request.headers.get("X-Engine-Token")
    if _engine_token and token != _engine_token:
        raise HTTPException(status_code=401, detail="token 无效")

class ProbeReq(BaseModel):
    videoPath: str

class JobReq(BaseModel):
    videoPath: str
    faceSwap: dict = Field(
        default_factory=lambda: {
            "enabled": False,
            "sourceImages": [],
            "restoreStrength": 0,
            "pasteRegion": "keep-eyes",
            "faceColorMatch": True,
        }
    )
    background: dict = Field(default_factory=lambda: {"mode": "keep"})
    options: dict = Field(default_factory=lambda: {"colorMatch": True, "quality": "balanced"})

_health_device: str | None = None

@app.get("/api/health")
async def health():
    """真实设备探测: 用小模型实际建一次会话, 报告生效的 provider。
    get_available_providers 只代表"可尝试", 会话可能因缺 DLL 静默回退 CPU。"""
    global _health_device
    settings = load_settings()
    if _health_device is None:
        try:
            from .models.base import create_session, session_device

            mp = resolve_model("scrfd_2.5g.onnx", settings)
            _health_device = session_device(create_session(str(mp), settings.get("device", "gpu")))
        except Exception:  # noqa: BLE001 (模型未下载等, 降级为设置值)
            _health_device = settings.get("device", "gpu")
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
    except Exception:  # noqa: BLE001
        providers = []
    return {
        "ok": True,
        "gpu": _health_device == "gpu",
        "device": settings.get("device", "gpu"),
        "activeDevice": _health_device,
        "providers": providers,
    }

@app.post("/api/probe", dependencies=[Depends(auth)])
async def probe_video(req: ProbeReq):
    try:
        return probe(req.videoPath)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e)) from e

@app.post("/api/jobs", dependencies=[Depends(auth)])
async def create_job(req: JobReq):
    # v1: 同一时间只跑一个任务,新任务排队
    running = [j for j in _jobs.values() if j.state in ("queued", "running")]
    if running:
        raise HTTPException(status_code=409, detail="已有任务在处理中,请等待完成或取消")

    req_dict = req.model_dump()
    req_dict["outputDir"] = load_settings().get("outputDir", "")
    job_id = uuid.uuid4().hex
    job = Job(job_id, req_dict, JOBS_ROOT)
    _jobs[job_id] = job
    _nudge_sd_unload()
    job.start()
    return {"jobId": job_id}

@app.get("/api/jobs/{job_id}", dependencies=[Depends(auth)])
async def job_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    return job.snapshot()

@app.post("/api/jobs/{job_id}/cancel", dependencies=[Depends(auth)])
async def cancel_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    job.cancel()
    return {"ok": True}

@app.get("/api/jobs/{job_id}/preview.jpg", dependencies=[Depends(auth)])
async def job_preview(job_id: str):
    job = _jobs.get(job_id)
    if not job or job.preview_jpeg is None:
        raise HTTPException(status_code=404, detail="暂无预览")
    return Response(content=job.preview_jpeg, media_type="image/jpeg")

@app.get("/api/thumb", dependencies=[Depends(auth)])
async def thumb(path: str):
    img = cv2.imread(path)
    if img is None:
        raise HTTPException(status_code=404, detail="图片无法读取")
    h, w = img.shape[:2]
    if max(h, w) > 96:
        scale = 96.0 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise HTTPException(status_code=500, detail="缩略图编码失败")
    return Response(content=buf.tobytes(), media_type="image/jpeg")

@app.get("/api/file", dependencies=[Depends(auth)])
async def serve_file(path: str):
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(str(p))

class SettingsModel(BaseModel):
    providers: list[dict] = []
    device: str = "gpu"
    modelsDir: str = ""
    outputDir: str = ""

@app.get("/api/settings", dependencies=[Depends(auth)])
async def get_settings():
    s = load_settings()
    if not s.get("outputDir"):
        s["outputDir"] = str(DATA_DIR.parent / "output")
    return s

@app.put("/api/settings", dependencies=[Depends(auth)])
async def put_settings(model: SettingsModel):
    return save_settings(model.model_dump())

@app.get("/api/models/status", dependencies=[Depends(auth)])
async def models_status():
    settings = load_settings()
    return {
        "files": downloads.status(settings["modelsDir"]),
        "progress": downloads.progress(),
    }

@app.post("/api/models/download", dependencies=[Depends(auth)])
async def models_download():
    downloads.download_all(str(downloads.target_dir()))
    return {"ok": True}

def main() -> None:
    import uvicorn

    port = int(os.environ.get("ENGINE_PORT", "8765"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

if __name__ == "__main__":
    main()
