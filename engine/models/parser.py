"""人脸分区解析 (BiSeNet, CelebAMask-HQ 19 类)。

用于贴回掩码: 只把生成脸贴进目标脸的指定分区 (皮肤/眉/鼻/嘴...),
眼镜与眼睛区域保留原帧, 解决"戴眼镜换脸脏"与贴回边界抖动。
分区-通道映射与 FaceFusion choices.face_mask_region_set 一致。
"""
import cv2
import numpy as np

# CelebAMask-HQ 标签通道 (0=背景; 7/8/9 为耳/饰等, 不参与贴回)
REGION_CHANNELS = {
    "skin": 1,
    "left-eyebrow": 2,
    "right-eyebrow": 3,
    "left-eye": 4,
    "right-eye": 5,
    "glasses": 6,
    "nose": 10,
    "mouth": 11,
    "upper-lip": 12,
    "lower-lip": 13,
}

# 镜片区不整块排除, 而是半透混合: 镜片后露出新脸眼睛, 消除"原脸眼睛 + 新脸皮肤"的拼贴感
# (已弃用半透方案: 半透会把原脸透出成重影, 改为目标眼镜区整体保留原帧压在新脸上)
GLASSES_ALPHA = 0.35

# 贴回范围预设 -> 纳入贴回掩码的分区
# full: 含眼睛(镜框走半透通道)
# keep-eyes: 同 full (眼睛区也换, 由椭圆外边界保证盖满, 原方案排除眼睛是拼贴感的来源之一)
# mid-face: 排除眉眼, 仅皮肤+鼻+嘴
REGION_PRESETS = {
    "full": ("skin", "left-eyebrow", "right-eyebrow", "left-eye", "right-eye", "nose", "mouth", "upper-lip", "lower-lip"),
    "keep-eyes": ("skin", "left-eyebrow", "right-eyebrow", "left-eye", "right-eye", "nose", "mouth", "upper-lip", "lower-lip"),
    "mid-face": ("skin", "nose", "mouth", "upper-lip", "lower-lip"),
}


class FaceParser:
    """512x512 输入, 输出 (512,512) uint8 标签图。"""

    INPUT_SIZE = 512

    def __init__(self, model_path: str, device: str = "gpu"):
        from .base import create_session, session_device

        self.session = create_session(model_path, device)
        self.device = session_device(self.session)
        self.input_name = self.session.get_inputs()[0].name

    def parse(self, crop_bgr: np.ndarray) -> np.ndarray:
        size = self.INPUT_SIZE
        img = cv2.resize(crop_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
        rgb = img[:, :, ::-1].astype(np.float32) / 255.0
        rgb -= np.array([0.485, 0.456, 0.406], dtype=np.float32)
        rgb /= np.array([0.229, 0.224, 0.225], dtype=np.float32)
        blob = rgb.transpose(2, 0, 1)[None].astype(np.float32)
        out = self.session.run(None, {self.input_name: blob})[0]
        return out[0].argmax(0).astype(np.uint8)


# 椭圆脸型掩码的 5 点模板系数 (相对对齐裁剪边长): 眉上/下颌下略外扩, 保证新脸盖满原脸
_OVAL_CX, _OVAL_CY, _OVAL_RX, _OVAL_RY = 0.5, 0.55, 0.46, 0.62


def oval_mask(size: int = 512) -> np.ndarray:
    """静态椭圆脸型掩码 (float32 0..1), 与 region mask 取并集保外沿。

    FaceFusion 同款思路: region mask 只删内部不该换的区, 外沿由 box/椭圆掩码
    保底, 否则按源脸脸型生成的新脸超出原脸皮肤区被裁掉, 露出原脸边缘成双层脸。
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    nx = (xx / size - _OVAL_CX) / _OVAL_RX
    ny = (yy / size - _OVAL_CY) / _OVAL_RY
    inside = (nx * nx + ny * ny) <= 1.0
    return inside.astype(np.float32)


def region_mask(labels: np.ndarray, preset: str) -> np.ndarray:
    """标签图 -> 指定预设的贴回掩码 (float32 0..1, 已羽化)。

    FaceFusion 同款后处理: 高斯模糊 sigma=5 后按 0.5 重映射, 边界平滑且稳定。
    """
    channels = [REGION_CHANNELS[r] for r in REGION_PRESETS.get(preset, REGION_PRESETS["keep-eyes"])]
    m = np.isin(labels, channels).astype(np.float32)
    m = cv2.GaussianBlur(m, (0, 0), 5)
    m = (m.clip(0.5, 1.0) - 0.5) * 2.0
    return m.clip(0.0, 1.0)


def mouth_inner_mask(labels: np.ndarray) -> np.ndarray:
    """嘴内区掩码 (mouth 通道减去上下唇): 标记 1 的区域贴回时保留原帧。

    inswapper 生成的嘴内 (牙齿/口腔) 常发灰或与原说话口型不符;
    FaceFusion 默认把嘴内排除在贴回外, 只换皮肤/唇, 口型连续性最好。
    """
    mouth = np.isin(labels, [REGION_CHANNELS["mouth"]]).astype(np.float32)
    lips = np.isin(
        labels, [REGION_CHANNELS["upper-lip"], REGION_CHANNELS["lower-lip"]]
    ).astype(np.float32)
    inner = mouth * (1.0 - lips)
    if inner.sum() < 30:  # 闭嘴/无明显嘴内区, 不启用
        return np.zeros_like(mouth)
    inner = cv2.erode(inner, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.GaussianBlur(inner, (0, 0), 2)


def glasses_mask(labels: np.ndarray) -> np.ndarray:
    """眼镜区掩码 (含镜框+镜片, 已羽化): 贴回时保留原帧, 用于源脸无镜场景。

    源脸无镜 + 目标有镜时, 生成脸不含镜片, 若把生成脸贴上去会"抹掉"眼镜;
    正确做法是把眼镜区从贴回掩码中整体挖掉, 原帧的镜框/镜片原样保留。
    """
    g = np.isin(labels, [REGION_CHANNELS["glasses"]]).astype(np.float32)
    if g.sum() < 50:
        return np.zeros_like(g)
    g = cv2.dilate(g, np.ones((5, 5), np.uint8), iterations=1)
    return cv2.GaussianBlur(g, (0, 0), 2)
