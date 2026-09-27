"""GFPGAN 1.4 面部修复 (ONNX)。"""
import cv2
import numpy as np


class FaceRestorer:
    def __init__(self, model_path: str, device: str = "gpu"):
        from .base import create_session

        self.session = create_session(model_path, device)
        self.input_name = self.session.get_inputs()[0].name
        size = self.session.get_inputs()[0].shape
        self.size = size[2] if isinstance(size[2], int) else 512

    def restore(self, face_bgr: np.ndarray) -> np.ndarray:
        """输入对齐后的人脸裁剪图,返回修复图。"""
        img = cv2.resize(face_bgr, (self.size, self.size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        blob = img.transpose(2, 0, 1)[None]
        out = self.session.run(None, {self.input_name: blob})[0][0]
        out = (out * 0.5 + 0.5).clip(0, 1).transpose(1, 2, 0)
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
