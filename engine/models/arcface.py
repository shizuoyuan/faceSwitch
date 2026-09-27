"""ArcFace 人脸识别 (arcface_w600k_r50, 512 维嵌入, 用于源脸匹配)。"""
import cv2
import numpy as np

from .base import similarity_transform

# ArcFace 112 参考关键点
ARC_REF = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def align_crop(frame_bgr: np.ndarray, kps: np.ndarray, size: int = 112) -> tuple[np.ndarray, np.ndarray]:
    """按 5 点关键点对齐裁剪,返回 (裁剪图, 2x3 仿射矩阵)。"""
    ref = ARC_REF * (size / 112.0)
    m = similarity_transform(kps, ref)
    aligned = cv2.warpAffine(frame_bgr, m, (size, size), borderValue=0)
    return aligned, m


class FaceRecognizer:
    def __init__(self, model_path: str, device: str = "gpu"):
        from .base import create_session

        self.session = create_session(model_path, device)
        self.input_name = self.session.get_inputs()[0].name

    def embedding(self, frame_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
        crop, _ = align_crop(frame_bgr, kps, 112)
        blob = cv2.dnn.blobFromImage(
            crop, 1.0 / 127.5, (112, 112), (127.5,) * 3, swapRB=True
        )
        emb = self.session.run(None, {self.input_name: blob})[0].reshape(-1)
        return emb / max(np.linalg.norm(emb), 1e-9)
