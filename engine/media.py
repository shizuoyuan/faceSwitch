"""视频探测 / ffmpeg 封装。"""
import base64
import subprocess
from pathlib import Path

import cv2
import numpy as np

try:
    import imageio_ffmpeg
    _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:  # noqa: BLE001
    _FFMPEG = "ffmpeg"


def ffmpeg_exe() -> str:
    return _FFMPEG


def _run_ffmpeg(args: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    cmd = [_FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def has_audio(video_path: str) -> bool:
    """imageio-ffmpeg 不带 ffprobe,用 ffmpeg 的流信息探测音频。"""
    cmd = [_FFMPEG, "-hide_banner", "-i", video_path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return "Audio:" in (r.stderr or "")


def probe(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # 缩略图取 10% 处的帧,避免黑屏开场
    if frame_count > 5:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_count * 0.1))
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    cap.release()

    if not ok:
        raise ValueError("无法读取视频帧")

    thumb = cv2.resize(frame, (width * 360 // max(height, 1), 360))
    ok2, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
    thumb_b64 = base64.b64encode(buf).decode() if ok2 else ""

    return {
        "path": video_path,
        "fileName": Path(video_path).name,
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "frameCount": frame_count,
        "durationSec": round(frame_count / fps, 2) if fps else 0,
        "hasAudio": has_audio(video_path),
        "thumbnail": f"data:image/jpeg;base64,{thumb_b64}",
    }


def extract_audio(video_path: str, out_path: str) -> bool:
    """抽取原音轨(无音频返回 False)。"""
    r = _run_ffmpeg(["-i", video_path, "-vn", "-c:a", "copy", out_path])
    return Path(out_path).exists() and r.returncode == 0


CRF_BY_QUALITY = {"high": 16, "balanced": 20, "fast": 26}


def encode_frames(
    frames_dir: str,
    fps: float,
    out_path: str,
    audio_path: str | None = None,
    quality: str = "balanced",
) -> str:
    frames_pattern = str(Path(frames_dir) / "f%06d.jpg")
    crf = CRF_BY_QUALITY.get(quality, 20)
    args = ["-framerate", str(fps), "-i", frames_pattern]
    if audio_path:
        args += ["-i", audio_path, "-c:a", "copy"]
    args += [
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
    ]
    if audio_path:
        args += ["-shortest"]
    args.append(out_path)
    r = _run_ffmpeg(args, timeout=3600)
    if r.returncode != 0:
        raise RuntimeError(f"编码失败: {r.stderr[-800:]}")
    return out_path


def fit_image(img: np.ndarray, w: int, h: int, mode: str = "cover") -> np.ndarray:
    """按 cover/contain/stretch 适配图像尺寸。"""
    if mode == "stretch":
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    ih, iw = img.shape[:2]
    if mode == "cover":
        scale = max(w / iw, h / ih)
    else:  # contain
        scale = min(w / iw, h / ih)
    nw, nh = max(int(iw * scale + 0.5), 1), max(int(ih * scale + 0.5), 1)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    x0 = max((w - nw) // 2, 0)
    y0 = max((h - nh) // 2, 0)
    x1, y1 = min(x0 + nw, w), min(y0 + nh, h)
    sx0, sy0 = x0 - min(x0, 0), y0 - min(y0, 0)
    canvas[y0:y1, x0:x1] = resized[sy0 : sy0 + (y1 - y0), sx0 : sx0 + (x1 - x0)]
    return canvas
