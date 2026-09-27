"""处理流水线: 单遍流式, 每帧依次经过 抠像 -> 背景 -> 色彩匹配 -> 换脸 -> 写出。

各阶段实现 Stage 协议 (setup/process/finish), 可独立替换。
"""
import base64
import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .config import load_settings
from .downloads import MODEL_MANIFEST
from .media import CRF_BY_QUALITY, encode_frames, extract_audio, ffmpeg_exe
from .models.arcface import FaceRecognizer
from .models.gfpgan import FaceRestorer
from .models.inswapper import FaceSwapper
from .models.rvm import VideoMatting
from .models.scrfd import FaceDetector
from .providers.remote import call_provider

STAGE_LABELS = {
    "prepare": "准备与抽帧",
    "matting": "人物抠像",
    "background": "背景处理",
    "harmonize": "光照融合",
    "faceswap": "AI 换脸",
    "compose": "合成与编码",
    "done": "完成",
}

# AI 生成背景的取景描述模板
_BG_PROMPT_TMPL = (
    "A photo-realistic empty scene background, wide angle, no people, "
    "cinematic lighting, high detail. Scene: {prompt}"
)


class JobError(Exception):
    pass


class Stage:
    name = "stage"

    def __init__(self, ctx: "JobContext"):
        self.ctx = ctx

    def setup(self) -> None:
        pass

    def process(self, frame: np.ndarray, index: int) -> np.ndarray:
        return frame

    def finish(self) -> None:
        pass


class JobContext:
    """跨阶段共享状态。"""

    def __init__(self, request: dict, workdir: Path):
        self.request = request
        self.workdir = workdir
        self.frames_dir = workdir / "frames"
        self.cancel = False
        self.error: str | None = None
        # RVM 首帧前生成的 AI 背景
        self.ai_background: np.ndarray | None = None


# ---------------------------------------------------------------- 抠像
class MattingStage(Stage):
    name = "matting"

    def setup(self) -> None:
        settings = load_settings()
        models_dir = settings["modelsDir"]
        self.ctx.rvm = VideoMatting(
            str(Path(models_dir) / "rvm_mobilenetv3_fp16.onnx"), settings["device"]
        )

    def process(self, frame: np.ndarray, index: int) -> np.ndarray:
        # 官方指南: 下采样分辨率落在 256~512 之间
        dr = float(np.clip(512.0 / max(frame.shape[:2]), 0.25, 0.6))
        pha, fgr, proc_size = self.ctx.rvm.process_frame(frame, dr)
        self.ctx.alpha = pha
        self.ctx.fgr = fgr
        self.ctx.proc_size = proc_size
        return frame


# ---------------------------------------------------------------- 背景
class BackgroundStage(Stage):
    name = "background"

    def setup(self) -> None:
        cfg = self.ctx.request["background"]
        self.mode = cfg["mode"]
        self.fit = cfg.get("fit", "cover")
        self.bg_img: np.ndarray | None = None
        self.bg_video: cv2.VideoCapture | None = None
        self.bg_frame_count = 0
        self.settings = load_settings()

        if self.mode == "image":
            self.bg_img = cv2.imread(cfg["imagePath"])
            if self.bg_img is None:
                raise JobError(f"背景图无法读取: {cfg['imagePath']}")
        elif self.mode == "video":
            self.bg_video = cv2.VideoCapture(cfg["videoPath"])
            if not self.bg_video.isOpened():
                raise JobError(f"背景视频无法打开: {cfg['videoPath']}")
            self.bg_frame_count = int(self.bg_video.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        elif self.mode == "ai":
            self.bg_img = None  # 首帧时生成

    def _gen_ai_background(self, first_frame: np.ndarray, w: int, h: int) -> np.ndarray:
        providers = self.settings.get("providers", [])
        provider = next(
            (p for p in providers if p["id"] == self.ctx.request["background"].get("providerId")),
            None,
        )
        if provider is None:
            raise JobError("未找到所选 AI 服务, 请检查设置")
        prompt = self.ctx.request["background"].get("prompt") or ""
        full_prompt = _BG_PROMPT_TMPL.format(prompt=prompt)
        first_b64 = ""
        if provider.get("bodyTemplate") and "{first_frame_b64}" in provider["bodyTemplate"]:
            ok, buf = cv2.imencode(".jpg", cv2.resize(first_frame, (640, 640 * first_frame.shape[0] // max(first_frame.shape[1], 1))))
            if ok:
                first_b64 = base64.b64encode(buf).decode()
        img = call_provider(provider, full_prompt, w, h, first_b64)
        from .media import fit_image

        return fit_image(img, w, h, self.fit)

    def process(self, frame: np.ndarray, index: int) -> np.ndarray:
        h, w = frame.shape[:2]
        if self.mode == "keep":
            return frame

        if self.mode == "ai" and self.bg_img is None:
            self.bg_img = self._gen_ai_background(frame, w, h)

        if self.mode == "video" and self.bg_video is not None:
            ok, bg = self.bg_video.read()
            if not ok or index >= self.bg_frame_count:
                self.bg_video.set(cv2.CAP_PROP_POS_FRAMES, index % max(self.bg_frame_count, 1))
                ok, bg = self.bg_video.read()
            if not ok:
                raise JobError("背景视频读取失败")
            self.bg_img = bg

        from .media import fit_image

        bg = fit_image(self.bg_img, w, h, self.fit)

        alpha = self.ctx.alpha
        ph, pw = self.ctx.proc_size
        fgr = self.ctx.fgr
        if (ph, pw) != (h, w):
            alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
            fgr = cv2.resize(fgr, (w, h), interpolation=cv2.INTER_LINEAR)
        alpha3 = alpha[..., None]

        # 前景来自原帧(细节保留), 边缘用 fgr 平滑过渡
        fg = frame.astype(np.float32)
        self.ctx.fg_bgr = fg
        out = bg.astype(np.float32) * (1 - alpha3) + fg * alpha3
        return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- 色彩匹配
class HarmonizeStage(Stage):
    """Reinhard 风格: 用背景统计量调整前景色彩, 统计量 EMA 平滑抗闪烁。"""

    name = "harmonize"

    def setup(self) -> None:
        self.enabled = self.ctx.request["options"]["colorMatch"]
        self.ema_mean: np.ndarray | None = None
        self.ema_std: np.ndarray | None = None

    def process(self, frame: np.ndarray, index: int) -> np.ndarray:
        if not self.enabled or self.mode_keep():
            return frame
        alpha = self.ctx.alpha
        h, w = frame.shape[:2]
        if alpha.shape != (h, w):
            alpha = cv2.resize(alpha, (w, h))
        mask = alpha > 0.4
        n = int(mask.sum())
        if n < 500:  # 无人或人太小, 不调整
            return frame

        bg = frame.astype(np.float32)
        bg_lab = cv2.cvtColor(bg.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)

        fg_lab = self.ctx.fg_bgr if hasattr(self.ctx, "fg_bgr") else bg
        # 用背景作为目标统计; 前景统计来自原帧人物区域
        src_lab = cv2.cvtColor(
            (self.ctx.fg_bgr.astype(np.uint8) if hasattr(self.ctx, "fg_bgr") else frame),
            cv2.COLOR_BGR2LAB,
        ).astype(np.float32)
        src_px = src_lab[mask]
        tgt_px = bg_lab.reshape(-1, 3)

        mean_s, std_s = src_px.mean(0), src_px.std(0) + 1e-5
        mean_t, std_t = tgt_px.mean(0), tgt_px.std(0) + 1e-5

        # 时序 EMA
        if self.ema_mean is None:
            self.ema_mean, self.ema_std = mean_s, std_s
        else:
            self.ema_mean = 0.85 * self.ema_mean + 0.15 * mean_s
            self.ema_std = 0.85 * self.ema_std + 0.15 * std_s

        adjusted = (src_lab - self.ema_mean) * (std_t / self.ema_std) + mean_t
        # 以人物区域的 alpha 混合调整结果
        a3 = cv2.GaussianBlur(alpha, (0, 0), 3)[..., None]
        out_lab = src_lab * (1 - a3) + adjusted * a3
        out_bgr = cv2.cvtColor(np.clip(out_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        out = frame * (1 - a3) + out_bgr * a3
        return np.clip(out, 0, 255).astype(np.uint8)

    def mode_keep(self) -> bool:
        return self.ctx.request["background"]["mode"] == "keep"


# ---------------------------------------------------------------- 换脸
class FaceSwapStage(Stage):
    name = "faceswap"

    def setup(self) -> None:
        cfg = self.ctx.request["faceSwap"]
        self.enabled = cfg["enabled"] and bool(cfg["sourceImages"])
        if not self.enabled:
            return
        settings = load_settings()
        models_dir = Path(settings["modelsDir"])
        device = settings["device"]

        self.detector = FaceDetector(str(models_dir / "scrfd_2.5g.onnx"), device)
        self.recognizer = FaceRecognizer(str(models_dir / "arcface_w600k_r50.onnx"), device)
        self.swapper = FaceSwapper(str(models_dir / "inswapper_128.onnx"), device)
        restore = int(cfg.get("restoreStrength", 0))
        self.restorer = (
            FaceRestorer(str(models_dir / "gfpgan_1.4.onnx"), device) if restore > 0 else None
        )
        self.restore_alpha = restore / 100.0

        self.source_embs: list[np.ndarray] = []
        for path in cfg["sourceImages"]:
            img = cv2.imread(path)
            if img is None:
                raise JobError(f"源脸图片无法读取: {path}")
            faces = self.detector.detect(img)
            if not faces:
                raise JobError(f"源脸图片中未检测到人脸: {path}")
            emb = self.recognizer.embedding(img, faces[0]["kps"])
            self.source_embs.append(emb)

        # 时序平滑: 上一帧各人脸的中心与关键点
        self.prev_tracks: list[dict] = []

    def process(self, frame: np.ndarray, index: int) -> np.ndarray:
        if not self.enabled:
            return frame
        faces = self.detector.detect(frame)
        out = frame

        for face in faces:
            kps = face["kps"]
            center = (face["bbox"][:2] + face["bbox"][2:]) / 2

            # 与上一帧最近的人脸做关键点 EMA, 抑制抖动与闪烁
            best, best_d = None, 1e9
            for tr in self.prev_tracks:
                d = np.linalg.norm(tr["center"] - center)
                if d < best_d:
                    best, best_d = tr, d
            diag = np.linalg.norm([frame.shape[1], frame.shape[0]])
            if best is not None and best_d < diag * 0.08:
                kps = 0.6 * kps + 0.4 * best["kps"]

            emb = self.recognizer.embedding(frame, kps)
            sims = [float(np.dot(emb, s)) for s in self.source_embs]
            src = self.source_embs[int(np.argmax(sims))]

            out = self.swapper.swap_into_frame(out, kps, src)

            if self.restorer is not None and self.restore_alpha > 0:
                from .models.arcface import align_crop

                crop, m = align_crop(out, kps, 128)
                restored = self.restorer.restore(crop)
                restored = cv2.resize(restored, (128, 128))
                blend = (
                    crop.astype(np.float32) * (1 - self.restore_alpha)
                    + restored.astype(np.float32) * self.restore_alpha
                ).astype(np.uint8)
                im = cv2.invertAffineTransform(m)
                h, w = out.shape[:2]
                warped = cv2.warpAffine(blend, im, (w, h), borderValue=0)
                mask = np.full((128, 128), 255, np.float32)
                mask = cv2.warpAffine(mask, im, (w, h), borderValue=0)
                mask = (mask > 20).astype(np.float32)
                mask = cv2.GaussianBlur(mask, (31, 31), 0)[..., None]
                out = (out * (1 - mask) + warped * mask).astype(np.uint8)

        # 更新轨迹
        self.prev_tracks = [
            {
                "center": (f["bbox"][:2] + f["bbox"][2:]) / 2,
                "kps": f["kps"],
            }
            for f in faces
        ]
        return out


# ---------------------------------------------------------------- 任务
class Job:
    def __init__(self, job_id: str, request: dict, jobs_root: Path):
        self.id = job_id
        self.request = request
        self.workdir = jobs_root / job_id
        self.state = "queued"
        self.stage = "prepare"
        self.progress = 0.0
        self.frame_index = 0
        self.frame_total = 0
        self.message = ""
        self.error: str | None = None
        self.output_path: str | None = None
        self.elapsed = 0.0
        self.preview_jpeg: bytes | None = None
        self.cancel_flag = False
        self._thread: threading.Thread | None = None

    def snapshot(self) -> dict:
        return {
            "id": self.id,
            "state": self.state,
            "stage": self.stage,
            "stageLabel": STAGE_LABELS.get(self.stage, self.stage),
            "progress": round(self.progress, 1),
            "frameIndex": self.frame_index,
            "frameTotal": self.frame_total,
            "message": self.message,
            "outputPath": self.output_path,
            "error": self.error,
            "elapsedSec": round(self.elapsed, 1),
        }

    def start(self) -> None:
        import threading

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        self.cancel_flag = True

    def _run(self) -> None:
        self.cancel_flag = False
        started = time.time()
        self.state = "running"
        self.stage = "prepare"
        self.message = ""
        try:
            self._pipeline()
            if self.cancel_flag:
                self.state = "canceled"
            else:
                self.state = "done"
                self.stage = "done"
                self.progress = 100.0
        except JobError as e:
            self.state, self.error = "failed", str(e)
        except Exception as e:  # noqa: BLE001
            import traceback

            self.state = "failed"
            self.error = f"{e}\n{traceback.format_exc()[-600:]}"
        finally:
            self.elapsed = time.time() - started

    def _pipeline(self) -> None:
        from .media import probe  # local import avoids cycle

        video_path = self.request["videoPath"]
        info = probe(video_path)
        fps = info["fps"]
        self.frame_total = info["frameCount"]

        if self.workdir.exists():
            shutil.rmtree(self.workdir, ignore_errors=True)
        self.frames_dir = self.workdir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        ctx = JobContext(self.request, self.workdir)
        ctx.alpha = None
        ctx.fgr = None
        ctx.proc_size = (0, 0)

        stages: list[Stage] = []
        need_matting = self.request["background"]["mode"] != "keep"
        if need_matting:
            stages.append(MattingStage(ctx))
            stages.append(BackgroundStage(ctx))
            stages.append(HarmonizeStage(ctx))
        if self.request["faceSwap"]["enabled"]:
            stages.append(FaceSwapStage(ctx))

        quality = self.request["options"]["quality"]

        # --- prepare: 抽音频 ---
        self.message = "正在抽取原音轨"
        audio_path = None
        if info["hasAudio"]:
            audio_path = str(self.workdir / "audio.m4a")
            if not extract_audio(video_path, audio_path):
                audio_path = None

        # --- 加载模型(计入 prepare 阶段) ---
        self.message = "正在加载 AI 模型"
        for st in stages:
            st.setup()
        self.progress = 2.0

        # --- 主循环: 单遍流式 ---
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise JobError(f"无法打开视频: {video_path}")
        out_idx = 0
        while True:
            if self.cancel_flag:
                cap.release()
                return
            ok, frame = cap.read()
            if not ok:
                break
            self.frame_index = out_idx
            for st in stages:
                self.stage = st.name
                frame = st.process(frame, out_idx)
            out_path = self.frames_dir / f"f{out_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if out_idx % 10 == 0:
                ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok2:
                    self.preview_jpeg = buf.tobytes()
            self.progress = 2.0 + 88.0 * (out_idx + 1) / max(self.frame_total, 1)
            out_idx += 1
        cap.release()

        if out_idx == 0:
            raise JobError("未能从视频中读取到任何帧")

        # --- 编码 ---
        self.stage = "compose"
        self.message = "正在合成视频"
        self.frame_total = out_idx
        out_name = f"faceswitch_{self.id[:8]}.mp4"
        out_path = Path(self.request.get("outputDir") or self.workdir) / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        encode_frames(
            str(self.frames_dir), fps, str(out_path), audio_path, quality
        )
        self.output_path = str(out_path)
        shutil.rmtree(self.frames_dir, ignore_errors=True)
