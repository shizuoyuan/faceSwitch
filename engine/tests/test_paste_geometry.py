"""贴回几何合成用例: 身份贴回 (fake=crop 自身) 后脸部区域应与原帧一致。"""
import sys
import numpy as np
import cv2

sys.path.insert(0, r"D:\face-switch\engine")
from models.arcface import align_crop, ARC_REF  # noqa: E402
from models.inswapper import FaceSwapper  # noqa: E402


def make_frame():
    # 低频渐变: 几何正确时往返插值损失应接近 0, 高频噪声会放大插值误差
    gx, gy = np.meshgrid(np.arange(720), np.arange(960))
    frame = np.zeros((960, 720, 3), np.float32)
    frame[:, :, 0] = (gx * 255) / 720
    frame[:, :, 1] = (gy * 255) / 960
    frame[:, :, 2] = ((gx + gy) * 255) / 1680
    return frame.astype(np.uint8)


def synth_kps(scale, center, angle_deg):
    """把 ARC_REF 模板放在指定尺码/位置/旋转上, 得到帧内关键点。"""
    theta = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    ref = ARC_REF - ARC_REF.mean(0)
    return (ref * scale) @ rot.T + np.array(center)


def check(swapper, scale, center=(360, 300), angle=-8.0):
    frame = make_frame()
    kps = synth_kps(scale, center, angle)
    crop, m = align_crop(frame, kps, 512)
    mask = np.ones((512, 512), np.float32)
    out = swapper.paste_into_frame(frame, crop, mask, m)

    # 贴回方形内部 (避开 8px 边缘的插值/羽化影响)
    corners = np.array([[8, 8, 1], [504, 8, 1], [504, 504, 1], [8, 504, 1]], np.float64)
    m_inv = cv2.invertAffineTransform(m.astype(np.float32)).astype(np.float64)
    pts = (m_inv @ corners.T).T[:, :2]
    x0, y0 = np.floor(pts.min(0)).astype(int)
    x1, y1 = np.ceil(pts.max(0)).astype(int)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, 719), min(y1, 959)
    diff = np.abs(out[y0:y1, x0:x1].astype(int) - frame[y0:y1, x0:x1].astype(int))
    return diff.mean(), diff.max(), (x0, y0, x1, y1)


if __name__ == "__main__":
    swapper = FaceSwapper.__new__(FaceSwapper)  # 不加载模型, 只测几何
    for scale in (3.2, 1.0, 0.5):  # 眼距≈161*scale px: 小脸/中脸/大脸
        mean_d, max_d, box = check(swapper, scale)
        status = "OK " if mean_d < 1.0 else "FAIL"
        print(f"[{status}] scale={scale}: mean_diff={mean_d:.2f} max_diff={max_d} box={box}")
