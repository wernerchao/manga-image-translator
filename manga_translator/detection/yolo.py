from __future__ import annotations

from typing import List

import cv2
import numpy as np
from pathlib import Path
import os

from .common import OfflineDetector
from ..utils import Quadrilateral
from ultralytics import YOLO

# Below this stddev an Otsu split is just carving up noise, not separating ink
# from background.
MIN_CROP_STDDEV = 8.0
# Plausible fraction of a textline box that is actually ink. Outside this range
# the polarity guess is almost certainly wrong.
MIN_INK_COVERAGE = 0.01
MAX_INK_COVERAGE = 0.9
BORDER_RING_FRACTION = 0.1


def _clip_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int):
    x1, y1, x2, y2 = np.round([x1, y1, x2, y2]).astype(np.int32)
    x1 = int(np.clip(x1, 0, width - 1))
    y1 = int(np.clip(y1, 0, height - 1))
    x2 = int(np.clip(x2, 0, width - 1))
    y2 = int(np.clip(y2, 0, height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _extract_ink_mask(gray: np.ndarray, x1: int, y1: int, x2: int, y2: int):
    """
    Binarize a textline box down to its ink pixels. Returns None when the crop
    cannot be split confidently, leaving it to the caller to fall back.
    """
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0 or crop.std() < MIN_CROP_STDDEV:
        return None

    _, bright = cv2.threshold(crop, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)

    # A detection box is tight around the textline, so its border ring is
    # background. Whichever class dominates the ring is therefore background,
    # and the other one is ink. This holds for dark-on-light bubble text and for
    # light-on-dark SFX alike, unlike picking the minority class by area, which
    # breaks on bold or oversized text.
    radius = max(1, int(round(min(crop.shape) * BORDER_RING_FRACTION)))
    ring = np.ones(crop.shape, dtype=bool)
    if min(crop.shape) > 2 * radius:
        ring[radius:-radius, radius:-radius] = False
    ink = cv2.bitwise_not(bright) if bright[ring].mean() > 127 else bright

    coverage = np.count_nonzero(ink) / ink.size
    if not MIN_INK_COVERAGE <= coverage <= MAX_INK_COVERAGE:
        return None
    return ink


class YoloDetector(OfflineDetector):
    """
    Text detector backed by a fine-tuned YOLO26l_animetext model.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = None
        self.device = "cpu"
        self.min_det_size = int(os.environ.get("yolo_min_det_size", 640))
        self.max_det_size = int(os.environ.get("yolo_max_det_size", 1280))
        self.textline_padding_px = int(os.environ.get("yolo_textline_padding_px", 8))
        # 'ink' binarizes each box down to its text pixels, 'box' fills the whole
        # box as the mask.
        self.mask_mode = os.environ.get("yolo_mask_mode", "ink").strip().lower()

    async def _load(self, device: str) -> None:
        self.device = device
        self.model = YOLO(
            os.path.join(os.path.dirname(__file__),"..","..","..","..","yolo_models", "yolo12l_animetext_finetuned_768_v2.1.pt")
        )
        self.model.to(self.device)

    async def _unload(self) -> None:
        self.model = None

    async def _infer(
        self,
        image: np.ndarray,
        detect_size: int,
        text_threshold: float,
        box_threshold: float,
        unclip_ratio: float,
        verbose: bool = False,
    ):
        height, width = image.shape[:2]
        raw_mask = np.zeros((height, width), dtype=np.uint8)
        image_size = (
            self.max_det_size
            if width > 1.5 * self.max_det_size or height > 1.5 * self.max_det_size
            else self.min_det_size
        )
        conf_threshold = float(np.clip(box_threshold, 0.0, 1.0))
        results = self.model.predict(
            source=image,
            imgsz=image_size,
            conf=conf_threshold,
            verbose=verbose,
            device=self.device,
        )

        if not results:
            return [], raw_mask, None

        result = results[0]
        if result.boxes is None or result.boxes.xyxy is None:
            return [], raw_mask, None

        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image

        textlines: List[Quadrilateral] = []
        for box, score in zip(boxes, scores):
            x1_f, y1_f, x2_f, y2_f = box.astype(np.float32)

            padded = _clip_box(
                x1_f - self.textline_padding_px,
                y1_f - self.textline_padding_px,
                x2_f + self.textline_padding_px,
                y2_f + self.textline_padding_px,
                width,
                height,
            )
            if padded is None:
                continue
            px1, py1, px2, py2 = padded

            pts = np.array(
                [
                    [px1, py1],
                    [px2, py1],
                    [px2, py2],
                    [px1, py2],
                ],
                dtype=np.int32,
            )

            quad = Quadrilateral(pts, "", float(score))
            if quad.area <= 16:
                continue

            textlines.append(quad)

            # The mask is seeded from the unpadded box: the padding exists so OCR
            # sees whole glyphs, but padded boxes of adjacent lines overlap, which
            # would fuse them into one blob here.
            ink = None
            unpadded = _clip_box(x1_f, y1_f, x2_f, y2_f, width, height)
            if unpadded is not None and self.mask_mode == "ink":
                ux1, uy1, ux2, uy2 = unpadded
                ink = _extract_ink_mask(gray, ux1, uy1, ux2, uy2)
                if ink is not None:
                    raw_mask[uy1:uy2, ux1:ux2] |= ink

            if ink is None:
                # Over-masking one box is recoverable; leaving its text
                # un-inpainted is not.
                cv2.fillPoly(raw_mask, [pts], 255)

        return textlines, raw_mask, None
