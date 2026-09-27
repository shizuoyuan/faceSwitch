"""inswapper_128 换脸: 对齐裁剪 -> 以源脸嵌入为条件生成 -> 掩码贴回。"""
import cv2
import numpy as np

from .arcface import align_crop


class FaceSwapper:
    def __init__(self, model_path: str, device: str = "gpu"):
        from .base import create_session

        self.session = create_session(model_path, device)
        self.input_name = self.session.get_inputs()[0].name
        self.src_name = self.session.get_inputs()[1].name

    def swap_into_frame(
        self,
        frame_bgr: np.ndarray,
        kps: np.ndarray,
        source_emb: np.ndarray,
    ) -> np.ndarray:
        """在整帧上对指定人脸执行换脸,返回新帧(不修改原图)。"""
        h, w = frame_bgr.shape[:2]
        crop, m = align_crop(frame_bgr, kps, 128)
        blob = cv2.dnn.blobFromImage(crop, 1.0 / 255.0, (128, 128), (0, 0, 0), swapRB=True)
        latent = source_emb.reshape(1, 512).astype(np.float32)
        pred = self.session.run(None, {self.input_name: blob, self.src_name: latent})[0]
        fake_rgb = pred.transpose(0, 2, 3, 1)[0]
        fake_bgr = np.clip(fake_rgb * 255.0, 0, 255).astype(np.uint8)[:, :, ::-1]
        return self._paste_back(frame_bgr, fake_bgr, m)

    def _paste_back(
        self, frame_bgr: np.ndarray, fake_bgr: np.ndarray, m: np.ndarray
    ) -> np.ndarray:
        """insightface 风格贴回: 人脸边界掩码 + 形态学收缩 + 高斯羽化。"""
        h, w = frame_bgr.shape[:2]
        crop, _ = align_crop(frame_bgr, self._kps_of_m(m), 128)

        diff = fake_bgr.astype(np.float32) - crop.astype(np.float32)
        diff = np.abs(diff).mean(axis=2)
        diff[:2, :] = diff[-2:, :] = 0
        diff[:, :2] = diff[:, -2:] = 0

        im = cv2.invertAffineTransform(m)
        fake_warp = cv2.warpAffine(fake_bgr, im, (w, h), borderValue=0)
        mask = np.full((128, 128), 255, dtype=np.float32)
        mask = cv2.warpAffine(mask, im, (w, h), borderValue=0)
        diff_warp = cv2.warpAffine(diff, im, (w, h), borderValue=0)

        mask[mask > 20] = 255
        diff_warp[diff_warp < 10] = 0
        diff_warp[diff_warp >= 10] = 255

        ys, xs = np.where(mask == 255)
        if len(ys) == 0:
            return frame_bgr
        mask_size = int(np.sqrt(ys.max() - ys.min()) * np.sqrt(xs.max() - xs.min()))
        k = max(mask_size // 10, 10)
        mask = cv2.erode(mask, np.ones((k, k), np.uint8), iterations=1)
        k2 = max(mask_size // 20, 5)
        mask = cv2.erode(mask, np.ones((k2, k2), np.uint8), iterations=1)
        mask = cv2.GaussianBlur(mask, (2 * k2 + 1, 2 * k2 + 1), 0)
        mask = (mask / 255.0)[..., None]

        out = mask * fake_warp.astype(np.float32) + (1 - mask) * frame_bgr.astype(np.float32)
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def _kps_of_m(m: np.ndarray) -> np.ndarray:
        """由仿射矩阵反推参考关键点(仅用于 paste_back 的 crop 对齐)。"""
        from .arcface import ARC_REF

        pts = np.concatenate([ARC_REF, np.ones((5, 1), np.float32)], axis=1)
        return (m @ pts.T).T
