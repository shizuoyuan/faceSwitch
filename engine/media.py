"""视频探测 / ffmpeg 封装。"""
import base64
import subprocess
import threading
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
        "fpsExact": fps,
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
    """按 cover/contain/stretch 适配图像尺寸。

    cover: 等比放大到铺满并居中裁掉出界部分; contain: 等比缩放到放得下, 居中留黑边。
    """
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
    # 居中摆放: 画布落点可为负 (cover 越界部分被裁), 源图起点随之偏移
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    sx0, sy0 = max(-x0, 0), max(-y0, 0)
    dx0, dy0 = max(x0, 0), max(y0, 0)
    dx1 = min(dx0 + nw - sx0, w)
    dy1 = min(dy0 + nh - sy0, h)
    canvas[dy0:dy1, dx0:dx1] = resized[sy0 : sy0 + (dy1 - dy0), sx0 : sx0 + (dx1 - dx0)]
    return canvas


class VideoFrames:
    """流式逐帧解码: ffmpeg -hwaccel auto 管道 → ffmpeg 软解管道 → OpenCV,三级降级。

    4K 视频用 OpenCV 软解会吃满 CPU; ffmpeg 管道可走 NVDec 等硬件解码。
    用法: `for frame in VideoFrames(path, w, h): ...`, 用完 close()。
    """

    _FRAME_BYTES: int

    def __init__(self, video_path: str, width: int, height: int):
        self.path = video_path
        self.width, self.height = width, height
        self._frame_bytes = width * height * 3
        self._proc: subprocess.Popen | None = None
        self._cap: cv2.VideoCapture | None = None
        self._mode: str | None = None  # None=未开始, hwaccel/ffmpeg/cv2=当前来源
        self.count = 0

    def _open_ffmpeg(self, hwaccel: bool) -> subprocess.Popen | None:
        args = ["-hide_banner", "-loglevel", "error"]
        if hwaccel:
            args += ["-hwaccel", "auto"]
        args += [
            "-nostdin", "-i", self.path,
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "pipe:1",
        ]
        try:
            return subprocess.Popen(
                [_FFMPEG, *args],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=self._frame_bytes,
            )
        except OSError:
            return None

    def _advance_source(self) -> bool:
        """切换到下一个解码来源; 全部失败返回 False。"""
        if self._mode is None:
            nxt = "hwaccel"
        elif self._mode == "hwaccel":
            nxt = "ffmpeg"
        elif self._mode == "ffmpeg":
            nxt = "cv2"
        else:
            return False
        self._mode = nxt
        if nxt == "cv2":
            cap = cv2.VideoCapture(self.path)
            if not cap.isOpened():
                return False
            self._cap = cap
            return True
        proc = self._open_ffmpeg(nxt == "hwaccel")
        if proc is None:
            return False
        self._proc = proc
        return True

    def __iter__(self) -> "VideoFrames":
        return self

    def __next__(self) -> np.ndarray:
        while True:
            if self._mode is None and not self._advance_source():
                raise ValueError(f"无法打开视频: {self.path}")

            if self._proc is not None:
                data = self._proc.stdout.read(self._frame_bytes)
                if len(data) == self._frame_bytes:
                    self.count += 1
                    return np.frombuffer(data, np.uint8).reshape(
                        self.height, self.width, 3
                    )
                # 流结束 (或解码器崩溃), 看 ffmpeg 退出码决定降级还是收尾
                rc = self._proc.wait()
                self._proc = None
                if rc != 0 and self._advance_source():
                    continue
                raise StopIteration

            if self._cap is not None:
                ok, frame = self._cap.read()
                if ok:
                    self.count += 1
                    return frame
                self._cap.release()
                self._cap = None
                raise StopIteration

            # 当前来源打不开, 尝试下一个
            if not self._advance_source():
                raise ValueError(f"无法打开视频: {self.path}")

    def close(self) -> None:
        if self._proc is not None:
            try:
                self._proc.stdout.close()
            except OSError:
                pass
            self._proc.kill()
            self._proc.wait()
            self._proc = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class VideoWriter:
    """帧数据直写 ffmpeg stdin (rawvideo) 单次编码, 音轨同命令合并。

    相比「逐帧 JPEG 落盘 -> ffmpeg 读图序列再编码」, 省掉双重编码与磁盘 IO。
    """

    def __init__(
        self,
        out_path: str,
        width: int,
        height: int,
        fps: float,
        audio_path: str | None = None,
        quality: str = "balanced",
    ):
        self.width, self.height = width, height
        crf = CRF_BY_QUALITY.get(quality, 20)
        args = [
            "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", _fmt_fps(fps), "-i", "pipe:0",
        ]
        if audio_path:
            args += ["-i", audio_path, "-c:a", "copy", "-shortest"]
        args += [
            "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path,
        ]
        self._proc = subprocess.Popen(
            [_FFMPEG, *args],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._stderr_tail = ""
        self._drain = threading.Thread(target=self._drain_stderr, daemon=True)
        self._drain.start()

    def _drain_stderr(self) -> None:
        # 不排空 stderr 管道会在缓冲区满时卡死 ffmpeg
        try:
            for line in self._proc.stderr:  # type: ignore[union-attr]
                self._stderr_tail = (self._stderr_tail + line.decode(errors="replace"))[-2000:]
        except (OSError, ValueError):
            pass

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            frame = cv2.resize(frame, (self.width, self.height))
        self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())  # type: ignore[union-attr]

    def close(self, timeout: int = 3600) -> None:
        try:
            self._proc.stdin.close()  # type: ignore[union-attr]
        except OSError:
            pass
        rc = self._proc.wait(timeout=timeout)  # type: ignore[union-attr]
        self._drain.join(timeout=5)
        if rc != 0:
            raise RuntimeError(f"编码失败: {self._stderr_tail[-800:]}")

    def abort(self) -> None:
        """中途放弃: 杀掉 ffmpeg, 不产出文件。"""
        try:
            self._proc.stdin.close()  # type: ignore[union-attr]
        except OSError:
            pass
        self._proc.kill()  # type: ignore[union-attr]
        self._proc.wait()  # type: ignore[union-attr]


def _fmt_fps(fps: float) -> str:
    """ffmpeg 帧率参数: 优先精确有理数, 避免长视频音画漂移。"""
    if abs(fps - round(fps)) < 1e-6:
        return str(round(fps))
    from fractions import Fraction

    fr = Fraction(fps).limit_denominator(1001)
    return f"{fr.numerator}/{fr.denominator}"
