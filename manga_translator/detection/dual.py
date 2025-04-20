import numpy as np
from typing import List, Tuple
import asyncio
import logging
import cv2
from shapely.geometry import Polygon

from .common import CommonDetector
from .default import DefaultDetector
from .ctd import ComicTextDetector
from ..utils import Quadrilateral

logger = logging.getLogger('manga_translator')

class DualDetector(CommonDetector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_detector = DefaultDetector(*args, **kwargs)
        self.ctd_detector = ComicTextDetector(*args, **kwargs)
        
        self.min_area = kwargs.get('min_area', 200)
        self.iou_threshold = kwargs.get('iou_threshold', 0.8)

    async def load(self, device: str):
        """Load models for both detectors."""
        logger.info(f"Loading models on device: {device}")
        await self.default_detector.load(device)
        await self.ctd_detector.load(device)
        logger.info("Both detector models loaded successfully")

    async def _detect(self, image: np.ndarray, detect_size: int, text_threshold: float, box_threshold: float,
                    unclip_ratio: float, verbose: bool = False) -> Tuple[List[Quadrilateral], np.ndarray, np.ndarray]:
        """Enhanced detection with prioritization for larger text boxes and overlap handling."""
        logger.info(f"Starting enhanced dual detection with image shape: {image.shape}")
        
        orig_h, orig_w = image.shape[:2]
        
        scale = detect_size / max(orig_h, orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        detection_params = {
            'detect_size': detect_size,
            'text_threshold': text_threshold,
            'box_threshold': box_threshold,
            'unclip_ratio': unclip_ratio,
            'invert': False,
            'gamma_correct': False,
            'rotate': False,
            'auto_rotate': False,
            'verbose': verbose
        }
        
        try:
            default_task = asyncio.create_task(self.default_detector.detect(resized_image, **detection_params))
            ctd_task = asyncio.create_task(self.ctd_detector.detect(resized_image, **detection_params))
            
            results = await asyncio.gather(default_task, ctd_task, return_exceptions=True)
            
            default_result, ctd_result = self._process_detector_results(results, resized_image, detection_params)
            
            if default_result is None and ctd_result is None:
                raise RuntimeError("Both detectors failed even after recovery attempts.")
            elif default_result is None:
                return self._process_single_detector_result(ctd_result, new_w, new_h, orig_w, orig_h)
            elif ctd_result is None:
                return self._process_single_detector_result(default_result, new_w, new_h, orig_w, orig_h)
            
            default_textlines, default_raw_mask, default_mask = default_result
            ctd_textlines, ctd_raw_mask, ctd_mask = ctd_result
            
            default_textlines = self._scale_quadrilaterals(default_textlines, new_w, new_h, orig_w, orig_h)
            ctd_textlines = self._scale_quadrilaterals(ctd_textlines, new_w, new_h, orig_w, orig_h)
            
            logger.info(f"After filtering: Default detector {len(default_textlines)} textlines, CTD {len(ctd_textlines)} textlines")
            
            merged_textlines = self._merge_and_prioritize_textlines(default_textlines, ctd_textlines, orig_w, orig_h)
            
            merged_raw_mask, merged_mask = self._create_merged_masks(
                default_textlines, ctd_textlines, 
                default_raw_mask, ctd_raw_mask, 
                default_mask, ctd_mask, 
                new_h, new_w
            )
            
            logger.info(f"Final merged result has {len(merged_textlines)} textlines")
                
            return merged_textlines, merged_raw_mask, merged_mask
            
        except Exception as e:
            logger.error(f"Unexpected error in dual detection: {str(e)}", exc_info=True)
            raise
    
    def _process_detector_results(self, results, resized_image, detection_params):
        """Process the results from both detectors and attempt recovery if needed."""
        default_result = results[0] if not isinstance(results[0], Exception) else None
        ctd_result = results[1] if not isinstance(results[1], Exception) else None
        
        if isinstance(results[0], Exception):
            logger.error(f"DefaultDetector failed: {str(results[0])}")
        if isinstance(results[1], Exception):
            logger.error(f"ComicTextDetector failed: {str(results[1])}")
        
        if default_result is None:
            try:
                recovery_params = detection_params.copy()
                recovery_params['gamma_correct'] = True
                default_result = asyncio.run(self.default_detector.detect(resized_image, **recovery_params))
                logger.info("DefaultDetector recovery succeeded with gamma correction")
            except Exception as e:
                logger.error(f"DefaultDetector recovery failed: {str(e)}")
        
        if ctd_result is None:
            try:
                recovery_params = detection_params.copy()
                recovery_params['gamma_correct'] = True
                recovery_params['text_threshold'] = max(0.3, detection_params['text_threshold'] - 0.1)
                ctd_result = asyncio.run(self.ctd_detector.detect(resized_image, **recovery_params))
                logger.info("ComicTextDetector recovery succeeded with adjusted parameters")
            except Exception as e:
                logger.error(f"ComicTextDetector recovery failed: {str(e)}")
                
        return default_result, ctd_result
    
    def _process_single_detector_result(self, result, new_w, new_h, orig_w, orig_h):
        """Process results when only one detector succeeded."""
        textlines, raw_mask, mask = result
        textlines = self._scale_quadrilaterals(textlines, new_w, new_h, orig_w, orig_h)
        return textlines, raw_mask, mask
    
    def _merge_and_prioritize_textlines(self, default_textlines: List[Quadrilateral], 
                                    ctd_textlines: List[Quadrilateral],
                                    img_w: int, img_h: int) -> List[Quadrilateral]:
        """Simplified merging with priority to default detector unless specific conditions are met."""
        # Start with all default textlines
        merged_textlines = default_textlines.copy()
        
        used_ctd_boxes = set()
        
        # Add CTD boxes that don't overlap with any default box
        for j, ctd_box in enumerate(ctd_textlines):
            if j in used_ctd_boxes:
                continue
                
            ctd_poly = Polygon(ctd_box.pts)
            overlaps = False
            
            for def_box in default_textlines:
                def_poly = Polygon(def_box.pts)
                
                if def_poly.intersects(ctd_poly):
                    try:
                        intersection_area = def_poly.intersection(ctd_poly).area
                        overlap_ratio = intersection_area / ctd_poly.area
                        
                        if overlap_ratio > 0.3:  # 30% overlap threshold
                            overlaps = True
                            break
                    except Exception as e:
                        logger.warning(f"Error calculating overlap: {str(e)}")
            
            if not overlaps:
                logger.info(f"Adding non-overlapping CTD box")
                merged_textlines.append(ctd_box)
        
        valid_textlines = []
        for tl in merged_textlines:
            try:
                poly = Polygon(tl.pts)
                if poly.is_valid and poly.area > self.min_area:
                    tl.area = poly.area
                    valid_textlines.append(tl)
            except Exception:
                continue
        
        kept_textlines = self._non_maximum_suppression(valid_textlines)
            
        return kept_textlines
        
    def _non_maximum_suppression(self, textlines: List[Quadrilateral]) -> List[Quadrilateral]:
        """Apply optimized non-maximum suppression to remove overlapping boxes."""
        if not textlines:
            return []
        
        polygons = [Polygon(tl.pts) for tl in textlines]
        areas = [poly.area for poly in polygons]
        
        indices = sorted(range(len(textlines)), key=lambda i: areas[i], reverse=True)
        kept_indices = []
        
        suppressed = [False] * len(textlines)
        
        for i, idx in enumerate(indices):
            if suppressed[idx]:
                continue
                
            kept_indices.append(idx)
            current_poly = polygons[idx]
            current_area = areas[idx]
            
            for j in range(i + 1, len(indices)):
                j_idx = indices[j]
                if suppressed[j_idx]:
                    continue
                    
                j_poly = polygons[j_idx]
                j_area = areas[j_idx]
                
                if not current_poly.envelope.intersects(j_poly.envelope):
                    continue
                    
                try:
                    intersection_area = current_poly.intersection(j_poly).area
                    union_area = current_area + j_area - intersection_area
                    iou = intersection_area / union_area if union_area > 0 else 0
                    
                    if iou > self.iou_threshold:
                        suppressed[j_idx] = True
                        
                except Exception as e:
                    logger.warning(f"Error in NMS intersection: {str(e)}")
        
        return [textlines[i] for i in kept_indices]
    
    def _create_merged_masks(self, default_textlines, ctd_textlines, 
                           default_raw_mask, ctd_raw_mask, 
                           default_mask, ctd_mask, 
                           new_h, new_w):
        """Create merged masks from both detectors based on weighted confidence."""
        default_confidence = np.mean([tl.prob for tl in default_textlines]) if default_textlines else 0
        ctd_confidence = np.mean([tl.prob for tl in ctd_textlines]) if ctd_textlines else 0
        total_confidence = default_confidence + ctd_confidence
        
        target_shape = (new_h, new_w)
        default_raw_mask = self._resize_mask(default_raw_mask, target_shape)
        ctd_raw_mask = self._resize_mask(ctd_raw_mask, target_shape)
        
        if default_mask is not None:
            default_mask = self._resize_mask(default_mask, target_shape)
        if ctd_mask is not None:
            ctd_mask = self._resize_mask(ctd_mask, target_shape)
        
        if total_confidence > 0:
            default_weight = default_confidence / total_confidence
            ctd_weight = ctd_confidence / total_confidence
            
            merged_raw_mask = default_weight * default_raw_mask + ctd_weight * ctd_raw_mask
            merged_raw_mask = (merged_raw_mask > 0.5).astype(np.uint8)
            
            if default_mask is not None and ctd_mask is not None:
                merged_mask = default_weight * default_mask + ctd_weight * ctd_mask
                merged_mask = (merged_mask > 0.5).astype(np.uint8)
            else:
                merged_mask = None
        else:
            merged_raw_mask = np.maximum(default_raw_mask, ctd_raw_mask)
            merged_mask = np.maximum(default_mask, ctd_mask) if default_mask is not None and ctd_mask is not None else None
        
        return merged_raw_mask, merged_mask
    
    def _resize_mask(self, mask, target_shape):
        """Resize mask to target shape."""
        if mask is None:
            return np.zeros(target_shape, dtype=np.uint8)
        
        if mask.shape != target_shape:
            return cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
        
        return mask
    
    def _scale_quadrilaterals(self, textlines: List[Quadrilateral], src_w: int, src_h: int, dst_w: int, dst_h: int) -> List[Quadrilateral]:
        """Scale quadrilateral coordinates from source resolution to destination resolution."""
        if not textlines:
            return textlines
        
        scale_x = dst_w / src_w
        scale_y = dst_h / src_h
        scaled_textlines = []
        
        for tl in textlines:
            scaled_pts = np.round(tl.pts * np.array([scale_x, scale_y])).astype(np.int32)
            scaled_tl = Quadrilateral(scaled_pts, tl.text, tl.prob)
            scaled_textlines.append(scaled_tl)
            
        return scaled_textlines