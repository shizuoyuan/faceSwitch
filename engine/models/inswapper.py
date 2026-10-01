"""inswapper_128 换脸: 对齐裁剪 -> 以源脸嵌入为条件生成 -> 由调用方掩码贴回。

贴回的掩码由 FaceSwapStage 提供 (人脸分区解析或差异掩码),
本模块只负责生成与 ROI 限制的几何贴回。
"""
import cv2
import numpy as np


class FaceSwapper:
    def __init__(self, model_path: str, device: str = "gpu"):
        from .base import create_session

        self.session = create_session(model_path, device)
        self.input_name = self.session.get_inputs()[0].name
        self.src_name = self.session.get_inputs()[1].name

    def generate(self, crop_bgr: np.ndarray, source_emb: np.ndarray) -> np.ndarray:
        """128 对齐裁剪 -> 生成 128 假脸 (不贴回)。"""
        blob = cv2.dnn.blobFromImage(crop_bgr, 1.0 / 255.0, (128, 128), (0, 0, 0), swapRB=True)
        latent = source_emb.reshape(1, 512).astype(np.float32)
        pred = self.session.run(None, {self.input_name: blob, self.src_name: latent})[0]
        fake_rgb = pred.transpose(0, 2, 3, 1)[0]
        return np.clip(fake_rgb * 255.0, 0, 255).astype(np.uint8)[:, :, ::-1]

    def paste_into_frame(
        self,
        frame_bgr: np.ndarray,
        fake_crop: np.ndarray,
        mask: np.ndarray,
        m: np.ndarray,
    ) -> np.ndarray:
        """把生成脸 crop (size,size) 连同 0..1 掩码按仿射矩阵 m 贴回整帧。

        全部 warp/混合限制在贴回区域包围盒内; 返回新帧, 不修改原图。
        """
        h, w = frame_bgr.shape[:2]
        size = fake_crop.shape[0]
        if mask.dtype != np.uint8:  # 0..1 float -> 0..255, warp 全程走 uint8
            mask = np.clip(mask * 255.0 + 0.5, 0, 255).astype(np.uint8)
        # m 映射 帧->裁剪画布, 贴回需取逆 (crop->帧), 与对齐 warp 同一套约定
        m_inv = cv2.invertAffineTransform(m)
        corners = np.array(
            [[0, 0, 1], [size, 0, 1], [size, size, 1], [0, size, 1]], np.float32
        )
        pts = (m_inv @ corners.T).T
        bx0, by0 = float(pts[:, 0].min()), float(pts[:, 1].min())
        bx1, by1 = float(pts[:, 0].max()), float(pts[:, 1].max())
        bw, bh = bx1 - bx0, by1 - by0
        pad = int(0.15 * max(bw, bh)) + 24  # 掩码羽化半径的富余量
        x0 = max(int(bx0) - pad, 0)
        y0 = max(int(by0) - pad, 0)
        x1 = min(int(bx1) + pad, w)
        y1 = min(int(by1) + pad, h)
        rw, rh = x1 - x0, y1 - y0
        if rw <= 0 or rh <= 0:
            return frame_bgr

        im_local = m_inv.copy()
        im_local[0, 2] -= x0
        im_local[1, 2] -= y0
        fake_warp = cv2.warpAffine(fake_crop, im_local, (rw, rh))
        mask_warp = cv2.warpAffine(mask, im_local, (rw, rh))

        # 定点混合 (7bit 权重): 全程 uint16 就地运算, 4K 大脸 ROI 下比 float32
        # 临时数组链少占 ~4/5 内存, 16GB 机器上不再触发 OOM
        out = frame_bgr.copy()
        roi = out[y0:y1, x0:x1]
        m16 = mask_warp.astype(np.uint16)  # 0..255
        # 混合点内移: 旧实现 w=mask/2, w=0.5 (mask=1.0) 才贴满, 羽化带内旧脸大面积
        # 透出 (发灰/重影)。改为 w = min(2*mask, 1): mask>=0.5 即贴满, 旧脸只在
        # 轮廓最外 mask<0.5 的窄带里淡出, 过渡带约 4-8px
        m9 = (np.minimum(m16 << 1, 255) >> 1)[..., None]  # 0..127
        r16 = roi.astype(np.uint16)
        r16 *= 128 - m9
        f16 = fake_warp.astype(np.uint16)
        f16 *= m9
        r16 += f16 + 64
        np.right_shift(r16, 7, out=r16)
        roi[...] = r16.astype(np.uint8)
        return out
