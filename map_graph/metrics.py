from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


def time_of_day_from_image(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.clip(gray.mean() / 255.0, 0.0, 1.0))


def overlap_ratio_from_offset(offset: Iterable[int], tile_size: Iterable[int]) -> float:
    dx, dy = [abs(int(v)) for v in offset]
    w, h = [int(v) for v in tile_size]
    if not np.isfinite(dx) or not np.isfinite(dy) or not np.isfinite(w) or not np.isfinite(h):
        return 0.0
    inter_w = max(0, w - dx)
    inter_h = max(0, h - dy)
    return float((inter_w * inter_h) / (w * h)) if w > 0 and h > 0 else 0.0


def coord_distance(a: Iterable[int], b: Iterable[int]) -> float:
    ax, ay = [int(v) for v in a]
    bx, by = [int(v) for v in b]
    return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)
