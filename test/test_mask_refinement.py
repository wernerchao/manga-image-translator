import cv2
import numpy as np
import pytest
from shapely.geometry import Polygon

from manga_translator.mask_refinement import text_mask_utils
from manga_translator.mask_refinement.text_mask_utils import complete_mask, extend_rect, refine_mask
from manga_translator.utils import Quadrilateral


def complete_mask_reference(img, mask, textlines, keep_threshold = 1e-2, dilation_offset = 0, kernel_size = 3):
    """
    Copy of `complete_mask` before the speed-up: whole-page bilateral filter and one shapely call
    per (component, textline) pair. The fast version must produce exactly the same mask.
    """
    bboxes = [txtln.aabb.xywh for txtln in textlines]
    polys = [Polygon(txtln.pts) for txtln in textlines]
    for (x, y, w, h) in bboxes:
        cv2.rectangle(mask, (x, y), (x + w, y + h), (0), 1)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    M = len(textlines)
    textline_ccs = [np.zeros_like(mask) for _ in range(M)]
    iinfo = np.iinfo(labels.dtype)
    textline_rects = np.full(shape = (M, 4), fill_value = [iinfo.max, iinfo.max, iinfo.min, iinfo.min], dtype = labels.dtype)
    ratio_mat = np.zeros(shape = (num_labels, M), dtype = np.float32)
    dist_mat = np.zeros(shape = (num_labels, M), dtype = np.float32)
    valid = False
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] <= 9:
            continue
        x1 = stats[label, cv2.CC_STAT_LEFT]
        y1 = stats[label, cv2.CC_STAT_TOP]
        w1 = stats[label, cv2.CC_STAT_WIDTH]
        h1 = stats[label, cv2.CC_STAT_HEIGHT]
        area1 = stats[label, cv2.CC_STAT_AREA]
        cc_pts = np.array([[x1, y1], [x1 + w1, y1], [x1 + w1, y1 + h1], [x1, y1 + h1]])
        cc_poly = Polygon(cc_pts)
        for tl_idx in range(M):
            area2 = polys[tl_idx].area
            overlapping_area = polys[tl_idx].intersection(cc_poly).area
            ratio_mat[label, tl_idx] = overlapping_area / min(area1, area2)
            dist_mat[label, tl_idx] = polys[tl_idx].distance(cc_poly.centroid)
        avg = np.argmax(ratio_mat[label])
        area2 = polys[avg].area
        if area1 >= area2:
            continue
        if ratio_mat[label, avg] <= keep_threshold:
            avg = np.argmin(dist_mat[label])
            area2 = polys[avg].area
            unit = max(min([textlines[avg].font_size, w1, h1]), 10)
            if dist_mat[label, avg] >= 0.5 * unit:
                continue
        textline_ccs[avg][y1:y1+h1, x1:x1+w1][labels[y1:y1+h1, x1:x1+w1] == label] = 255
        textline_rects[avg, 0] = min(textline_rects[avg, 0], x1)
        textline_rects[avg, 1] = min(textline_rects[avg, 1], y1)
        textline_rects[avg, 2] = max(textline_rects[avg, 2], x1 + w1)
        textline_rects[avg, 3] = max(textline_rects[avg, 3], y1 + h1)
        valid = True

    if not valid:
        return None

    textline_rects[:, 2] -= textline_rects[:, 0]
    textline_rects[:, 3] -= textline_rects[:, 1]

    final_mask = np.zeros_like(mask)
    img = cv2.bilateralFilter(img, 17, 80, 80)
    for i, cc in enumerate(textline_ccs):
        x1, y1, w1, h1 = textline_rects[i]
        text_size = min(w1, h1, textlines[i].font_size)
        x1, y1, w1, h1 = extend_rect(x1, y1, w1, h1, img.shape[1], img.shape[0], int(text_size * 0.1))
        dilate_size = max((int((text_size + dilation_offset) * 0.3) // 2) * 2 + 1, 3)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        cc_region = np.ascontiguousarray(cc[y1: y1 + h1, x1: x1 + w1])
        if cc_region.size == 0:
            continue
        img_region = np.ascontiguousarray(img[y1: y1 + h1, x1: x1 + w1])
        cc_region = refine_mask(img_region, cc_region)
        cc[y1: y1 + h1, x1: x1 + w1] = cc_region
        x2, y2, w2, h2 = extend_rect(x1, y1, w1, h1, img.shape[1], img.shape[0], -(-dilate_size // 2))
        cc[y2:y2+h2, x2:x2+w2] = cv2.dilate(cc[y2:y2+h2, x2:x2+w2], kern)
        final_mask[y2:y2+h2, x2:x2+w2] = cv2.bitwise_or(final_mask[y2:y2+h2, x2:x2+w2], cc[y2:y2+h2, x2:x2+w2])
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(final_mask, kern)


def draw_text_line(img, x, y, n_chars, char_size, vertical = False):
    """Draws a row (or column) of glyph-like ink blobs and returns the line's corner points."""
    step = int(char_size * 1.2)
    for c in range(n_chars):
        cx = x + (0 if vertical else c * step)
        cy = y + (c * step if vertical else 0)
        cv2.rectangle(img, (cx + 2, cy + 2), (cx + char_size - 2, cy + 5), (20, 20, 20), -1)
        cv2.line(img, (cx + char_size // 2, cy + 2), (cx + char_size // 2, cy + char_size - 2), (30, 30, 30), 3)
        cv2.circle(img, (cx + char_size // 3, cy + char_size * 2 // 3), max(char_size // 6, 2), (10, 10, 10), -1)
    w = char_size if vertical else n_chars * step
    h = n_chars * step if vertical else char_size
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def make_page(seed, width = 420, height = 360):
    rng = np.random.default_rng(seed)
    img = np.full((height, width, 3), 235, dtype = np.uint8)
    img = np.clip(img.astype(np.int16) + rng.integers(-20, 20, img.shape), 0, 255).astype(np.uint8)
    lines = [
        draw_text_line(img, 40, 30, 8, 24),
        draw_text_line(img, 40, 70, 6, 20),
        draw_text_line(img, 300, 60, 6, 22, vertical = True),
        # Touches the left and bottom page edges
        draw_text_line(img, 0, height - 26, 5, 26),
        # Touches the right and top page edges
        draw_text_line(img, width - 24, 0, 4, 24, vertical = True),
    ]
    # A tilted line, plus a line box with no ink inside it at all
    lines.append([[60, 200], [220, 170], [226, 200], [66, 230]])
    cv2.fillPoly(img, [np.array([[80, 205], [200, 182], [203, 192], [83, 215]])], (15, 15, 15))
    lines.append([[150, 120], [260, 120], [260, 150], [150, 150]])
    # Stray ink far from every line: must not join any line
    cv2.circle(img, (380, 300), 6, (0, 0, 0), -1)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = np.where(gray < 100, 255, 0).astype(np.uint8)
    textlines = [Quadrilateral(np.array(pts, dtype = np.float64), '', 1) for pts in lines]
    return img, mask, textlines


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_complete_mask_matches_reference(seed):
    img, mask, textlines = make_page(seed)

    expected = complete_mask_reference(img.copy(), mask.copy(), textlines)
    actual = complete_mask(img.copy(), mask.copy(), textlines)

    assert expected is not None
    assert expected.any()
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('rect', [
    (0, 0, 50, 40),        # page corner
    (370, 320, 49, 39),    # opposite corner, ends on the last pixel
    (100, 100, 5, 3),      # smaller than the filter
    (200, 10, 120, 300),   # tall
])
def test_bilateral_filter_regions_matches_whole_image(rect):
    img, _, _ = make_page(3)
    x, y, w, h = rect
    expected = cv2.bilateralFilter(img, 17, 80, 80)
    actual = text_mask_utils.bilateral_filter_regions(img, [rect, (0, 0, 0, 10), (10, 10, -5, 4)])
    np.testing.assert_array_equal(actual[y:y+h, x:x+w], expected[y:y+h, x:x+w])
    outside = np.ones(img.shape[:2], dtype = bool)
    outside[y:y+h, x:x+w] = False
    assert not actual[outside].any()
