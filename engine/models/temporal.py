"""关键点时序滤波 (One-Euro)。

自适应平滑: 静止时强滤抖动, 快速运动时弱滤跟手。
替代一阶 EMA, 消除检测帧与外推帧之间的关键点回跳。
"""
import numpy as np


class OneEuroKps:
    """向量化 One-Euro 滤波, 作用于 (N, 2) 关键点数组。"""

    def __init__(self, freq: float = 30.0, min_cutoff: float = 1.2, beta: float = 0.08, d_cutoff: float = 1.2):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x: np.ndarray | None = None
        self.dx: np.ndarray | None = None

    @staticmethod
    def _alpha(cutoff, freq: float):
        # cutoff 可为标量或与 x 同形数组 (逐点自适应)
        tau = 1.0 / (2.0 * np.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        if self.x is None:
            self.x = x.copy()
            self.dx = np.zeros_like(x)
            return self.x.copy()
        new_dx = (x - self.x) * self.freq
        a_d = self._alpha(self.d_cutoff, self.freq)
        self.dx = a_d * new_dx + (1.0 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        a = self._alpha(cutoff, self.freq)
        self.x = a * x + (1.0 - a) * self.x
        return self.x.copy()
