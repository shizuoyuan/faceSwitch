"""SCRFD 人脸检测 (scrfd_2.5g, 输入 640, 含 5 点关键点)。"""
import cv2
import numpy as np

from .base import nms, resize_letterbox

STRIDES = [8, 16, 32]
INPUT_SIZE = 640
DET_THRESHOLD = 0.5


class FaceDetector:
    def __init__(self, model_path: str, device: str = "gpu"):
        from .base import create_session

        self.session = create_session(model_path, device)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        # onnx 固定尺寸或动态都按 640 处理
        self.input_w = inp.shape[3] if isinstance(inp.shape[3], int) else INPUT_SIZE
        self.input_h = inp.shape[2] if isinstance(inp.shape[2], int) else INPUT_SIZE

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        h, w = frame_bgr.shape[:2]
        det_img, scale = resize_letterbox(frame_bgr, self.input_w)
        blob = cv2.dnn.blobFromImage(
            det_img, 1.0 / 128.0, (self.input_w, self.input_h), (127.5,) * 3, swapRB=True
        )
        outputs = self.session.run(None, {self.input_name: blob})

        # 9 输出: 3 组 (score, bbox, kps),按 stride 顺序
        fmc = len(outputs) // 3
        all_boxes, all_scores, all_kps = [], [], []
        for idx, stride in enumerate(STRIDES[:fmc]):
            scores = outputs[idx].reshape(-1)
            bbox_preds = outputs[idx + fmc].reshape(-1, 4) * stride
            kps_preds = outputs[idx + fmc * 2].reshape(-1, 10) * stride

            fmap_h, fmap_w = self.input_h // stride, self.input_w // stride
            n_anchors = scores.size // (fmap_h * fmap_w)

            grid = np.stack(np.mgrid[:fmap_h, :fmap_w][::-1], axis=-1).astype(np.float32)
            centers = (grid * stride).reshape(-1, 2)
            if n_anchors > 1:
                centers = np.stack([centers] * n_anchors, axis=1).reshape(-1, 2)

            # distance2bbox
            x1 = centers[:, 0] - bbox_preds[:, 0]
            y1 = centers[:, 1] - bbox_preds[:, 1]
            x2 = centers[:, 0] + bbox_preds[:, 2]
            y2 = centers[:, 1] + bbox_preds[:, 3]
            boxes = np.stack([x1, y1, x2, y2], axis=-1)

            kps = np.empty((len(centers), 10), dtype=np.float32)
            for j in range(5):
                kps[:, 2 * j] = centers[:, 0] + kps_preds[:, 2 * j]
                kps[:, 2 * j + 1] = centers[:, 1] + kps_preds[:, 2 * j + 1]

            pos = np.where(scores > DET_THRESHOLD)[0]
            all_boxes.append(boxes[pos])
            all_scores.append(scores[pos])
            all_kps.append(kps[pos])

        if not all_boxes:
            return []
        boxes = np.concatenate(all_boxes)
        scores = np.concatenate(all_scores)
        kpss = np.concatenate(all_kps)

        keep = nms(boxes, scores, 0.4)
        faces = []
        for i in keep:
            bx = boxes[i] / scale
            kps = kpss[i].reshape(5, 2) / scale
            bx[[0, 2]] = bx[[0, 2]].clip(0, w - 1)
            bx[[1, 3]] = bx[[1, 3]].clip(0, h - 1)
            faces.append(
                {"bbox": bx, "kps": kps, "score": float(scores[i])}
            )
        faces.sort(key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]), reverse=True)
        return faces
