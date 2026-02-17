from __future__ import annotations

from typing import List

import cv2
import numpy as np

from .common import OfflineDetector
from ..utils import Quadrilateral


class YoloDetector(OfflineDetector):
    """
    Drop-in text detector backed by a YOLO26s model.
    """

    _MODEL_MAPPING = {}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = None
        self.device = "cpu"

    async def _load(self, device: str) -> None:
        self.device = device
        self.model = None

        try:
            from ultralytics import YOLO

            self.model = YOLO("yolo26s.pt")
            try:
                self.model.to(self.device)
            except Exception as error:
                self.logger.warning(
                    "Could not move YOLO26s model to device '%s': %s",
                    self.device,
                    error,
                )
        except Exception as error:
            self.logger.warning(
                "Failed to initialize YOLO26s model. " "Detector will run as no-op: %s",
                error,
            )

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
        self.logger.info(
            "YOLO _infer called: detect_size=%s text_threshold=%.4f box_threshold=%.4f unclip_ratio=%.4f",
            detect_size,
            text_threshold,
            box_threshold,
            unclip_ratio,
        )
        height, width = image.shape[:2]
        raw_mask = np.zeros((height, width), dtype=np.uint8)

        if self.model is None:
            return [], raw_mask, None

        conf_threshold = float(np.clip(box_threshold, 0.0, 1.0))
        image_size = int(max(detect_size, 32))

        results = self.model.predict(
            source=image,
            # imgsz=image_size,
            conf=conf_threshold,
            verbose=verbose,
            device=self.device,
            text=["text region"],
        )

        if not results:
            return [], raw_mask, None

        result = results[0]
        if result.boxes is None or result.boxes.xyxy is None:
            return [], raw_mask, None

        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()

        textlines: List[Quadrilateral] = []
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = np.round(box).astype(np.int32)
            x1 = int(np.clip(x1, 0, width - 1))
            y1 = int(np.clip(y1, 0, height - 1))
            x2 = int(np.clip(x2, 0, width - 1))
            y2 = int(np.clip(y2, 0, height - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            pts = np.array(
                [
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2],
                ],
                dtype=np.int32,
            )

            quad = Quadrilateral(pts, "", float(score))
            if quad.area <= 16:
                continue

            textlines.append(quad)
            cv2.fillPoly(raw_mask, [pts], 255)

        return textlines, raw_mask, None
