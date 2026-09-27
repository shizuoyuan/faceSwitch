"""RVM 视频人像抠像 (ONNX 循环模型, 时序天然稳定)。"""
import cv2
import numpy as np


class VideoMatting:
    """逐帧推理;内部维护循环隐状态。输出 alpha (H,W) float32 与前景 BGR。"""

    def __init__(self, model_path: str, device: str = "gpu", max_side: int = 1280):
        from .base import create_session, session_device

        self.session = create_session(model_path, device)
        self.device = session_device(self.session)
        self.max_side = max_side

        self.input_names = [i.name for i in self.session.get_inputs()]
        self.state_in = [n for n in self.input_names if n.endswith("i") and n != "src"]
        self.state_in.sort()  # r1i, r2i, ...
        self.is_fp16 = self.session.get_inputs()[0].type == "tensor(float16)"

        # RVM 导出模型内置归一化,输入取 0-255 RGB。
        self._states: dict[str, np.ndarray] = {}

    def _zero_states(self, dtype: np.dtype) -> dict[str, np.ndarray]:
        return {n: np.zeros((1, 1, 1, 1), dtype=dtype) for n in self.state_in}

    def _prep(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        h, w = frame_bgr.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        if scale < 1.0:
            frame = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
        else:
            frame = frame_bgr
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        src = rgb.transpose(2, 0, 1)[None]
        if self.is_fp16:
            src = src.astype(np.float16)
        return src, scale, h, w

    def process_frame(self, frame_bgr: np.ndarray, downsample_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 (alpha float32 HxW 0..1 [原图尺寸], fgr_bgr uint8 [处理尺寸], 处理图)。"""
        src, scale, oh, ow = self._prep(frame_bgr)
        dtype = np.float16 if self.is_fp16 else np.float32
        if not self._states:
            self._states = self._zero_states(dtype)

        feed: dict[str, np.ndarray] = {"src": src}
        for n in self.state_in:
            feed[n] = self._states[n]
        dr_name = "downsample_ratio"
        if dr_name in self.input_names:
            # 官方要求 downsample_ratio 恒为 FP32
            feed[dr_name] = np.asarray([downsample_ratio], dtype=np.float32)

        outputs = self.session.run(None, feed)
        named = dict(zip([o.name for o in self.session.get_outputs()], outputs))
        pha = named.get("pha")
        fgr = named.get("fgr")

        for n in self.state_in:
            self._states[n] = named[n.replace("i", "o", 1)]

        pha = pha[0, 0].astype(np.float32)
        fgr_bgr = cv2.cvtColor(
            (fgr[0].transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
        )
        proc_h, proc_w = pha.shape
        if (proc_h, proc_w) != (oh, ow) and scale != 1.0:
            pha = cv2.resize(pha, (ow, oh), interpolation=cv2.INTER_LINEAR)
        return pha, fgr_bgr, (proc_h, proc_w)
