"""处理流水线: 单遍流式, 每帧依次经过 抠像 -> 背景 -> 色彩匹配 -> 换脸 -> 写出。

各阶段实现 Stage 协议 (setup/process/finish), 可独立替换。

性能设计 (4K 基准):
- 解码走 ffmpeg (优先硬件加速), 帧数据直写 ffmpeg stdin 编码, 无中间 JPEG 落盘
- 默认三段流水线并行: 解码线程 -> [抠像+背景+调色 (有状态, 保序)] -> [换脸+写出];
  GPU 推理与 CPU 合成重叠, 有界队列限制内存占用
- FACESWITCH_SERIAL=1 环境变量强制串行 (回归逃生开关)
- 人脸检测隔帧执行 (options.detectInterval, 默认 4), 帧间用关键点速度外推
"""
import base64
import os
import queue
import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .config import load_settings, resolve_model
from .downloads import MODEL_MANIFEST
from .media import (
    VideoFrames,
    VideoWriter,
    extract_audio,
    fit_image,
)
from .models.arcface import FaceRecognizer, align_crop
from .models.gfpgan import FaceRestorer
from .models.inswapper import FaceSwapper
from .models.parser import (
    REGION_CHANNELS,
    FaceParser,
    glasses_mask,
    mouth_inner_mask,
    oval_mask,
    region_mask,
)
from .models.rvm import VideoMatting
from .models.scrfd import FaceDetector
from .models.temporal import OneEuroKps
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

TIMING_LABELS = {
    "prepare": "准备",
    "decode": "解码",
    "matting": "抠像",
    "background": "背景",
    "harmonize": "融合",
    "faceswap": "换脸",
    "write": "写帧",
    "encode": "收尾编码",
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


class Timings:
    """各环节累计耗时 (线程安全: 各 key 只被单一线程写入, GIL 下 dict 操作原子)。"""

    def __init__(self):
        self._t: dict[str, float] = {}

    def add(self, key: str, sec: float) -> None:
        self._t[key] = self._t.get(key, 0.0) + sec

    def snapshot(self) -> dict[str, float]:
        return {k: round(v, 1) for k, v in sorted(self._t.items(), key=lambda x: -x[1])}


def _person_roi(alpha: np.ndarray, h: int, w: int, threshold: float, margin: int = 32):
    """人物包围盒 (外扩 margin); 无人返回 None。"""
    ys, xs = np.nonzero(alpha > threshold)
    if ys.size == 0:
        return None
    y0 = max(int(ys.min()) - margin, 0)
    y1 = min(int(ys.max()) + 1 + margin, h)
    x0 = max(int(xs.min()) - margin, 0)
    x1 = min(int(xs.max()) + 1 + margin, w)
    return y0, y1, x0, x1


class JobContext:
    """跨阶段共享状态。"""

    def __init__(self, request: dict, workdir: Path):
        self.request = request
        self.workdir = workdir
        self.cancel = False
        self.error: str | None = None
        self.exc: BaseException | None = None
        # RVM 首帧前生成的 AI 背景
        self.ai_background: np.ndarray | None = None
        # 帧间共享: RVM 输出与原帧
        self.alpha: np.ndarray | None = None  # uint8 0..255, 帧尺寸
        self.fgr: np.ndarray | None = None
        self.proc_size = (0, 0)
        self.fg_frame: np.ndarray | None = None  # 合成前的原帧 (uint8 引用, 不复制)


# ---------------------------------------------------------------- 抠像
class MattingStage(Stage):
    name = "matting"

    def setup(self) -> None:
        settings = load_settings()
        self.ctx.rvm = VideoMatting(
            str(resolve_model("rvm_mobilenetv3_fp16.onnx", settings)), settings["device"]
        )

    def process(self, frame: np.ndarray, index: int) -> np.ndarray:
        # 官方指南: 下采样分辨率落在 256~512 之间
        dr = float(np.clip(512.0 / max(frame.shape[:2]), 0.25, 0.6))
        pha, fgr, proc_size = self.ctx.rvm.process_frame(frame, dr)
        self.ctx.alpha = pha  # uint8 0..255, 帧尺寸
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
        self._fit_cache: tuple[tuple[int, int], np.ndarray] | None = None
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

        return fit_image(img, w, h, self.fit)

    def _fitted_bg(self, w: int, h: int) -> np.ndarray:
        """静态背景 (image/ai) 的 fit 结果缓存, video 模式每帧 fit。"""
        if self.mode in ("image", "ai"):
            if self._fit_cache is not None and self._fit_cache[0] == (w, h):
                return self._fit_cache[1]
            fitted = fit_image(self.bg_img, w, h, self.fit)
            self._fit_cache = ((w, h), fitted)
            return fitted
        return fit_image(self.bg_img, w, h, self.fit)

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

        out = self._fitted_bg(w, h).copy()
        alpha = self.ctx.alpha
        self.ctx.fg_frame = frame

        # float32 混合只在人物 ROI 内做, ROI 外直接用背景 (4K 下省 60-90% 像素工作量)
        roi = _person_roi(alpha, h, w, 2)  # 2/255 ≈ 0.008
        if roi is None:
            return out
        y0, y1, x0, x1 = roi
        a = alpha[y0:y1, x0:x1, None].astype(np.float32) * (1.0 / 255.0)
        out_roi = out[y0:y1, x0:x1].astype(np.float32)
        out_roi *= 1 - a
        out_roi += frame[y0:y1, x0:x1].astype(np.float32) * a
        out[y0:y1, x0:x1] = np.clip(out_roi, 0, 255).astype(np.uint8)
        return out


# ---------------------------------------------------------------- 色彩匹配
class HarmonizeStage(Stage):
    """Reinhard 风格: 用背景统计量调整前景色彩, 统计量 EMA 平滑抗闪烁。

    全部运算限制在人物 ROI 内; 调色用每通道 LUT 代替全帧 float32 乘加,
    数学与原实现逐通道等价 (量化到 uint8 栅格)。
    """

    name = "harmonize"

    def setup(self) -> None:
        self.enabled = self.ctx.request["options"]["colorMatch"]
        self.ema_mean: np.ndarray | None = None
        self.ema_std: np.ndarray | None = None

    def process(self, frame: np.ndarray, index: int) -> np.ndarray:
        if not self.enabled or self.mode_keep():
            return frame
        h, w = frame.shape[:2]
        alpha = self.ctx.alpha
        if alpha is None:
            return frame
        roi = _person_roi(alpha, h, w, 102)  # 102/255 ≈ 0.4
        if roi is None:
            return frame
        y0, y1, x0, x1 = roi

        src_frame = self.ctx.fg_frame if self.ctx.fg_frame is not None else frame
        src_lab = cv2.cvtColor(src_frame[y0:y1, x0:x1], cv2.COLOR_BGR2LAB)  # uint8
        mask = alpha[y0:y1, x0:x1] > 102
        px = src_lab[mask][::2]  # 子采样算统计量
        if px.shape[0] < 500:
            return frame
        mean_s, std_s = px.mean(0), px.std(0) + 1e-5

        # 目标统计: 合成帧全图 LAB (uint8 转换, 降采样)
        tgt_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        tgt_px = tgt_lab[::4, ::4].reshape(-1, 3)
        mean_t, std_t = tgt_px.mean(0), tgt_px.std(0) + 1e-5

        # 时序 EMA
        if self.ema_mean is None:
            self.ema_mean, self.ema_std = mean_s, std_s
        else:
            self.ema_mean = 0.85 * self.ema_mean + 0.15 * mean_s
            self.ema_std = 0.85 * self.ema_std + 0.15 * std_s

        # 每通道仿射 -> LUT: (x - ema_mean) * (std_t / ema_std) + mean_t
        lut = np.stack(
            [
                np.clip(
                    (np.arange(256) - self.ema_mean[c]) * (std_t[c] / self.ema_std[c]) + mean_t[c],
                    0,
                    255,
                ).astype(np.uint8)
                for c in range(3)
            ],
            axis=1,
        )[None, ...]
        a = (
            cv2.GaussianBlur(alpha[y0:y1, x0:x1], (0, 0), 3).astype(np.float32) * (1.0 / 255.0)
        )[..., None]
        adjusted = cv2.LUT(src_lab, lut)

        out_lab = src_lab.astype(np.float32) * (1 - a) + adjusted.astype(np.float32) * a
        out_bgr = cv2.cvtColor(np.clip(out_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        frame_roi = frame[y0:y1, x0:x1].astype(np.float32)
        frame_roi *= 1 - a
        frame_roi += out_bgr.astype(np.float32) * a
        frame[y0:y1, x0:x1] = np.clip(frame_roi, 0, 255).astype(np.uint8)
        return frame

    def mode_keep(self) -> bool:
        return self.ctx.request["background"]["mode"] == "keep"


# ---------------------------------------------------------------- 换脸
def _diff_mask(fake512: np.ndarray, crop512: np.ndarray) -> np.ndarray:
    """解析模型不可用时的回退掩码: 生成脸与原脸的差异区域。"""
    diff = np.abs(fake512.astype(np.float32) - crop512.astype(np.float32)).mean(2)
    diff[:16, :] = diff[-16:, :] = 0
    diff[:, :16] = diff[:, -16:] = 0
    mask = (diff > 10).astype(np.float32)
    mask = cv2.erode(mask, np.ones((9, 9), np.uint8), iterations=1)
    mask = cv2.GaussianBlur(mask, (0, 0), 5)
    return mask.clip(0, 1)


class FaceSwapStage(Stage):
    name = "faceswap"

    def setup(self) -> None:
        cfg = self.ctx.request["faceSwap"]
        self.enabled = cfg["enabled"] and bool(cfg["sourceImages"])
        if not self.enabled:
            return
        settings = load_settings()
        device = settings["device"]

        self.detector = FaceDetector(str(resolve_model("scrfd_2.5g.onnx", settings)), device)
        self.recognizer = FaceRecognizer(
            str(resolve_model("arcface_w600k_r50.onnx", settings)), device
        )
        self.swapper = FaceSwapper(str(resolve_model("inswapper_128.onnx", settings)), device)
        restore = int(cfg.get("restoreStrength", 0))
        self.restorer = (
            FaceRestorer(str(resolve_model("gfpgan_1.4.onnx", settings)), device)
            if restore > 0
            else None
        )
        self.restore_alpha = restore / 100.0

        # 贴回范围预设与换脸颜色匹配
        self.paste_region = cfg.get("pasteRegion", "keep-eyes")
        self.face_color_match = bool(cfg.get("faceColorMatch", True))
        # 人脸分区解析 (可选模型): 缺失时回退差异掩码, 不阻塞任务
        self.parser = None
        try:
            self.parser = FaceParser(
                str(resolve_model("bisenet_resnet_34.onnx", settings)), device
            )
        except Exception:  # noqa: BLE001
            self.parser = None
        # 静态椭圆脸型掩码: 与 region mask 取并集保外沿, 防止新脸超出原脸
        # 皮肤区的部分被裁掉后露出原脸边缘 (双层脸的根源)
        self.oval512 = oval_mask(512)
        # 源脸是否戴眼镜: 决定目标眼镜区的处理方式 (保留原帧 vs 随新脸替换)
        self.src_has_glasses = False
        self.source_embs: list[np.ndarray] = []
        for path in cfg["sourceImages"]:
            img = cv2.imread(path)
            if img is None:
                raise JobError(f"源脸图片无法读取: {path}")
            faces = self.detector.detect(img)
            if not faces:
                raise JobError(f"源脸图片中未检测到人脸: {path}")
            if self.parser is not None:
                crop512, _ = align_crop(img, faces[0]["kps"], 512)
                g = np.isin(
                    self.parser.parse(crop512), [REGION_CHANNELS["glasses"]]
                ).astype(np.float32)
                self.src_has_glasses = self.src_has_glasses or g.sum() > 50
            emb = self.recognizer.embedding(img, faces[0]["kps"])
            self.source_embs.append(emb)

        # 时序状态: 上一帧各人脸的中心/关键点/滤波器/颜色 EMA
        self.prev_tracks: list[dict] = []
        # 检测解耦: 每 N 帧检测一次, 帧间用关键点速度外推
        self.detect_every = max(
            1, int((self.ctx.request.get("options") or {}).get("detectInterval", 4))
        )
        self._cache: dict = {"index": -10**9, "faces": []}
        self._last_src: np.ndarray | None = None

    def _detect(self, frame: np.ndarray, index: int) -> list[dict]:
        faces = self.detector.detect(frame)
        diag = float(np.linalg.norm([frame.shape[1], frame.shape[0]]))
        prev_faces = self._cache["faces"] if self._cache["faces"] else []
        dt = max(index - self._cache["index"], 1)
        for f in faces:
            center = (f["bbox"][:2] + f["bbox"][2:]) / 2
            f["center"] = center
            best, best_d = None, 1e9
            for pf in prev_faces:
                d = float(np.linalg.norm(pf["center"] - center))
                if d < best_d:
                    best, best_d = pf, d
            if best is not None and best_d < diag * 0.08:
                f["vel"] = (f["kps"] - best["kps"]) / dt
            else:
                f["vel"] = np.zeros_like(f["kps"])
        self._cache = {"index": index, "faces": faces}
        return faces

    def _predicted(self, frame: np.ndarray, index: int) -> list[dict]:
        dt = index - self._cache["index"]
        h, w = frame.shape[:2]
        out = []
        for f in self._cache["faces"]:
            kps = f["kps"] + f["vel"] * dt
            kps[:, 0] = np.clip(kps[:, 0], 0, w - 1)
            kps[:, 1] = np.clip(kps[:, 1], 0, h - 1)
            shift = f["vel"].mean(0) * dt  # (2,)
            out.append(
                {
                    "kps": kps,
                    "bbox": f["bbox"] + np.concatenate([shift, shift]),
                    "score": f["score"],
                    "center": (f["bbox"][:2] + f["bbox"][2:]) / 2 + shift,
                }
            )
        return out

    def _color_match(self, state: dict, fake512: np.ndarray, crop512: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """把生成脸的 LAB 统计量对齐到目标脸; 目标统计量按轨迹 EMA 平滑防闪烁。"""
        m = mask > 0.5
        if int(m.sum()) < 200:
            return fake512
        fake_lab = cv2.cvtColor(fake512, cv2.COLOR_BGR2LAB)
        tgt_lab = cv2.cvtColor(crop512, cv2.COLOR_BGR2LAB)
        mean_f = fake_lab[m].mean(0)
        std_f = fake_lab[m].std(0) + 1e-5
        mean_t = tgt_lab[m].mean(0)
        std_t = tgt_lab[m].std(0) + 1e-5
        prev_mean, prev_std = state.get("cm_mean"), state.get("cm_std")
        if prev_mean is None:
            t_mean, t_std = mean_t, std_t
        else:
            t_mean = 0.85 * prev_mean + 0.15 * mean_t
            t_std = 0.85 * prev_std + 0.15 * std_t
        state["cm_mean"], state["cm_std"] = t_mean, t_std
        adj = (fake_lab.astype(np.float32) - mean_f) * (t_std / std_f) + t_mean
        return cv2.cvtColor(np.clip(adj, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    def process(self, frame: np.ndarray, index: int) -> np.ndarray:
        if not self.enabled:
            return frame
        detect_now = (index - self._cache["index"]) >= self.detect_every
        if detect_now:
            faces = self._detect(frame, index)
        else:
            faces = self._predicted(frame, index)
        out = frame
        single_src = len(self.source_embs) == 1
        diag = np.linalg.norm([frame.shape[1], frame.shape[0]])
        new_tracks: list[dict] = []

        for face in faces:
            center = (face["bbox"][:2] + face["bbox"][2:]) / 2

            # 轨迹匹配 -> One-Euro 滤波 (静止强滤抖动, 快动弱滤跟手)
            best, best_d = None, 1e9
            for tr in self.prev_tracks:
                d = np.linalg.norm(tr["center"] - center)
                if d < best_d:
                    best, best_d = tr, d
            if best is not None and best_d < diag * 0.08:
                filt = best["filter"]
                cm_state = best.get("cm") or {}
            else:
                filt = OneEuroKps()
                cm_state = {}
            kps = filt(face["kps"])

            # 源脸选择: 单源直接用; 多源仅检测帧识别, 隔帧沿用轨迹上的结果
            if single_src:
                src = self.source_embs[0]
            elif best is not None and best.get("emb") is not None and not detect_now:
                src = best["emb"]
            else:
                emb = self.recognizer.embedding(frame, kps)
                sims = [float(np.dot(emb, s)) for s in self.source_embs]
                src = self.source_embs[int(np.argmax(sims))]
            self._last_src = src

            # 生成 128 假脸, 在 512 坐标系里修复/调色/掩码/贴回
            crop128, _ = align_crop(frame, kps, 128)
            fake128 = self.swapper.generate(crop128, src)
            crop512, m512 = align_crop(frame, kps, 512)

            # 修复前移: GFPGAN 直接在生成阶段出 512 (原实现贴回后二次裁剪, 清晰度受限)
            if self.restorer is not None and self.restore_alpha > 0:
                restored = self.restorer.restore(fake128)
                base512 = cv2.resize(fake128, (512, 512), interpolation=cv2.INTER_CUBIC)
                fake512 = (
                    base512.astype(np.float32) * (1 - self.restore_alpha)
                    + restored.astype(np.float32) * self.restore_alpha
                ).astype(np.uint8)
                # 保护眼镜区: GFPGAN 会把镜框当脸修导致变形, 镜框区保留未修复底图
                if self.parser is not None:
                    gmask = np.isin(
                        self.parser.parse(crop512), [REGION_CHANNELS["glasses"]]
                    ).astype(np.float32)
                    if gmask.any():
                        gmask = cv2.GaussianBlur(gmask, (0, 0), 3)[..., None]
                        fake512 = (
                            fake512.astype(np.float32) * (1 - gmask)
                            + base512.astype(np.float32) * gmask
                        ).astype(np.uint8)
            else:
                fake512 = cv2.resize(fake128, (512, 512), interpolation=cv2.INTER_CUBIC)

            # 贴回掩码: 椭圆脸型保外沿 ∪ 分区解析删内部; 解析模型不可用时回退差异掩码
            if self.parser is not None:
                tgt_labels = self.parser.parse(crop512)
                mask = region_mask(tgt_labels, self.paste_region)
                mask = np.maximum(mask, self.oval512)
                # 假脸轮廓掩码: 对生成脸跑解析, 贴回范围精确等于新脸轮廓,
                # 消除"新脸脸型与原脸不一致导致的双层/错位感"
                fake_labels = self.parser.parse(fake512)
                fake_skin = np.isin(
                    fake_labels,
                    [
                        REGION_CHANNELS["skin"],
                        REGION_CHANNELS["left-eyebrow"],
                        REGION_CHANNELS["right-eyebrow"],
                        REGION_CHANNELS["left-eye"],
                        REGION_CHANNELS["right-eye"],
                        REGION_CHANNELS["nose"],
                        REGION_CHANNELS["mouth"],
                        REGION_CHANNELS["upper-lip"],
                        REGION_CHANNELS["lower-lip"],
                    ],
                ).astype(np.float32)
                fake_skin = cv2.dilate(fake_skin, np.ones((7, 7), np.uint8), iterations=1)
                fake_skin = cv2.GaussianBlur(fake_skin, (0, 0), 5)
                fake_skin = (fake_skin.clip(0.5, 1.0) - 0.5) * 2.0
                mask = np.minimum(mask, fake_skin.clip(0.0, 1.0))
                # 嘴内区保留原帧 (FaceFusion 同款): 牙齿/口腔不替换, 口型与原视频连续
                keep = mouth_inner_mask(tgt_labels)
                if keep.any():
                    mask = np.minimum(mask, 1.0 - keep)
                # 眼镜: 源脸无镜 + 目标有镜 -> 镜区整体保留原帧, 避免生成脸"抹掉"眼镜;
                # 源脸有镜 -> 生成脸自带镜片, 随新脸正常贴回
                if not self.src_has_glasses:
                    gm = glasses_mask(tgt_labels)
                    if gm.any():
                        mask = np.minimum(mask, 1.0 - gm)
                # 羽化: σ10 过渡带 ~30px 旧脸透出重影, σ5+0.5截断只有 ~10px 且
                # 截断让梯度中段出现斜率断层, 肉眼仍读出"贴纸边"。
                # 折中 σ8 + 全程 smoothstep (两端二阶平滑、无斜率断层), 过渡带 ~18px;
                # 内部实心区锁回 1, 防分区掩码内部接缝跌破阈值被裁成洞
                mask = cv2.GaussianBlur(mask, (0, 0), 8)
                t = mask.clip(0.0, 1.0)
                mask = t * t * (3.0 - 2.0 * t)
                core = cv2.erode(
                    (t >= 0.98).astype(np.uint8), np.ones((9, 9), np.uint8), iterations=1
                )
                mask = np.maximum(mask, core.astype(np.float32))
            else:
                mask = _diff_mask(fake512, crop512)

            if self.face_color_match:
                fake512 = self._color_match(cm_state, fake512, crop512, mask)

            if self.parser is not None:
                # 接缝低频过渡: 全局色彩匹配仍可能留下边界处的亮度/色阶跳变。
                # 取裁剪图与生成脸的低频差 (大半径模糊) 补回 fake —— 贴回处 alpha=mask,
                # seam 附近 alpha≈0.5 时若按 (1-mask) 加权, 颜色差只补一半, 残留色阶;
                # 故权重按 2.2×edge 放大并截断到 1, 使过渡带内颜色差基本补满,
                # 脸心 (mask=1, edge=0) 不受影响。半径 31 覆盖约 2 倍羽化带宽度
                edge = (1.0 - mask).astype(np.float32)
                w = (edge * 2.2).clip(0.0, 1.0)[..., None]
                low = cv2.GaussianBlur(
                    crop512.astype(np.float32) - fake512.astype(np.float32), (0, 0), 31
                )
                fake512 = np.clip(
                    fake512.astype(np.float32) + low * w, 0, 255
                ).astype(np.uint8)

            out = self.swapper.paste_into_frame(out, fake512, mask, m512)

            new_tracks.append(
                {
                    "center": center,
                    "kps": kps,
                    "emb": self._track_emb(detect_now),
                    "filter": filt,
                    "cm": cm_state,
                }
            )

        self.prev_tracks = new_tracks
        return out

    def _track_emb(self, detect_now: bool) -> np.ndarray | None:
        """轨迹上记录该脸对应的源脸 embedding (多源 + 隔帧检测时使用)。"""
        if len(self.source_embs) == 1:
            return self.source_embs[0]
        if detect_now:
            return self._last_src
        return None


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
        self.timings = Timings()
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
            "stageTimingsSec": self.timings.snapshot(),
        }

    def start(self) -> None:
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

    # ---- 帧写出 (含预览) ----
    def _write_frame(self, writer: VideoWriter, frame: np.ndarray, idx: int) -> None:
        writer.write(frame)
        if idx % 10 == 0:
            # 预览缩到 960 宽, 避免 4K 全幅 JPEG 编码开销
            h, w = frame.shape[:2]
            if w > 960:
                frame = cv2.resize(frame, (960, h * 960 // w), interpolation=cv2.INTER_AREA)
            ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok2:
                self.preview_jpeg = buf.tobytes()

    # ---- 串行路径 (FACESWITCH_SERIAL=1, 回归逃生开关) ----
    def _run_serial(self, ctx: JobContext, stages: list[Stage], decoder, writer: VideoWriter) -> None:
        for idx, frame in enumerate(decoder):
            if self.cancel_flag:
                return
            self.frame_index = idx
            for st in stages:
                self.stage = st.name
                t0 = time.perf_counter()
                frame = st.process(frame, idx)
                self.timings.add(st.name, time.perf_counter() - t0)
            t0 = time.perf_counter()
            self._write_frame(writer, frame, idx)
            self.timings.add("write", time.perf_counter() - t0)
            self.progress = 2.0 + 88.0 * (idx + 1) / max(self.frame_total, 1)

    # ---- 并行路径 (默认) ----
    def _run_parallel(self, ctx: JobContext, stages: list[Stage], decoder, writer: VideoWriter) -> None:
        q_raw: queue.Queue = queue.Queue(maxsize=3)
        q_core: queue.Queue = queue.Queue(maxsize=3)
        core_stages = [st for st in stages if st.name in ("matting", "background", "harmonize")]
        out_stages = [st for st in stages if st.name == "faceswap"]

        def _put(q: queue.Queue, item) -> bool:
            while True:
                if self.cancel_flag or ctx.exc is not None:
                    return False
                try:
                    q.put(item, timeout=0.5)
                    return True
                except queue.Full:
                    continue

        def _get(q: queue.Queue):
            while True:
                try:
                    item = q.get(timeout=0.5)
                    return None if item is None else item
                except queue.Empty:
                    if self.cancel_flag or ctx.exc is not None:
                        return None

        def decode_worker() -> None:
            try:
                for idx, frame in enumerate(decoder):
                    t0 = time.perf_counter()
                    ok = _put(q_raw, (idx, frame))
                    self.timings.add("decode", time.perf_counter() - t0)
                    if not ok:
                        return
            except Exception as e:  # noqa: BLE001
                ctx.exc = e
            finally:
                _put(q_raw, None)

        def core_worker() -> None:
            try:
                while True:
                    item = _get(q_raw)
                    if item is None:
                        return
                    idx, frame = item
                    for st in core_stages:
                        self.stage = st.name
                        t0 = time.perf_counter()
                        frame = st.process(frame, idx)
                        self.timings.add(st.name, time.perf_counter() - t0)
                    if not _put(q_core, (idx, frame)):
                        return
            except Exception as e:  # noqa: BLE001
                ctx.exc = e
            finally:
                _put(q_core, None)

        def out_worker() -> None:
            try:
                while True:
                    item = _get(q_core)
                    if item is None:
                        return
                    idx, frame = item
                    for st in out_stages:
                        self.stage = st.name
                        t0 = time.perf_counter()
                        frame = st.process(frame, idx)
                        self.timings.add(st.name, time.perf_counter() - t0)
                    t0 = time.perf_counter()
                    self._write_frame(writer, frame, idx)
                    self.timings.add("write", time.perf_counter() - t0)
                    self.frame_index = idx
                    self.progress = 2.0 + 88.0 * (idx + 1) / max(self.frame_total, 1)
            except Exception as e:  # noqa: BLE001
                ctx.exc = e

        threads = [
            threading.Thread(target=decode_worker, name="decode", daemon=True),
            threading.Thread(target=core_worker, name="core", daemon=True),
            threading.Thread(target=out_worker, name="out", daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if ctx.exc is not None and not self.cancel_flag:
            raise ctx.exc

    def _pipeline(self) -> None:
        from .media import probe  # local import avoids cycle

        video_path = self.request["videoPath"]
        info = probe(video_path)
        fps = info.get("fpsExact") or info["fps"]
        self.frame_total = info["frameCount"]

        if self.workdir.exists():
            shutil.rmtree(self.workdir, ignore_errors=True)
        self.workdir.mkdir(parents=True, exist_ok=True)

        ctx = JobContext(self.request, self.workdir)

        stages: list[Stage] = []
        need_matting = self.request["background"]["mode"] != "keep"
        if need_matting:
            stages.append(MattingStage(ctx))
            stages.append(BackgroundStage(ctx))
            stages.append(HarmonizeStage(ctx))
        if self.request["faceSwap"]["enabled"]:
            stages.append(FaceSwapStage(ctx))

        quality = self.request["options"]["quality"]

        # --- prepare: 抽音频 + 加载模型 ---
        t0 = time.perf_counter()
        self.message = "正在抽取原音轨"
        audio_path = None
        if info["hasAudio"]:
            audio_path = str(self.workdir / "audio.m4a")
            if not extract_audio(video_path, audio_path):
                audio_path = None

        self.message = "正在加载 AI 模型"
        for st in stages:
            st.setup()
        self.timings.add("prepare", time.perf_counter() - t0)
        self.progress = 2.0

        out_name = f"faceswitch_{self.id[:8]}.mp4"
        out_path = Path(self.request.get("outputDir") or self.workdir) / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        decoder = VideoFrames(video_path, info["width"], info["height"])
        writer = VideoWriter(str(out_path), info["width"], info["height"], fps, audio_path, quality)
        try:
            try:
                serial = os.environ.get("FACESWITCH_SERIAL") == "1"
                if serial:
                    self._run_serial(ctx, stages, decoder, writer)
                else:
                    self._run_parallel(ctx, stages, decoder, writer)
            finally:
                decoder.close()

            if self.frame_index == 0 and not self.cancel_flag:
                raise JobError("未能从视频中读取到任何帧")

            if self.cancel_flag or ctx.exc is not None:
                writer.abort()
                if ctx.exc is not None:
                    raise ctx.exc
                return

            # --- 编码收尾 (帧数据已在写帧时流入 ffmpeg) ---
            self.stage = "compose"
            self.message = "正在合成视频"
            self.frame_total = self.frame_index + 1
            t0 = time.perf_counter()
            writer.close()
            self.timings.add("encode", time.perf_counter() - t0)
            self.output_path = str(out_path)
        except BaseException:
            # 任一环节异常/取消: 杀掉 ffmpeg, 不产出半成品文件
            writer.abort()
            raise
