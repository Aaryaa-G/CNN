"""Standard CenterNet-style target encoding helpers (gaussian heatmap radius
and splatting) and box decoding from a predicted heatmap/wh/offset triplet."""
import numpy as np
import torch


def gaussian_radius(box_w, box_h, min_overlap=0.7):
    """Radius such that a gaussian of that radius, offset by up to `radius`
    from the true center, still overlaps the box by >= min_overlap IoU.
    Standard CornerNet/CenterNet formula (min of three quadratic-formula cases)."""
    a1 = 1
    b1 = box_h + box_w
    c1 = box_w * box_h * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(max(b1 ** 2 - 4 * a1 * c1, 0))
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (box_h + box_w)
    c2 = (1 - min_overlap) * box_w * box_h
    sq2 = np.sqrt(max(b2 ** 2 - 4 * a2 * c2, 0))
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (box_h + box_w)
    c3 = (min_overlap - 1) * box_w * box_h
    sq3 = np.sqrt(max(b3 ** 2 - 4 * a3 * c3, 0))
    r3 = (b3 + sq3) / 2

    return max(min(r1, r2, r3), 0)


def draw_gaussian(heatmap, center_xy, radius):
    """In-place max-splat a 2D gaussian onto heatmap (H, W) at center_xy=(cx,cy)."""
    radius = max(int(radius), 0)
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    h, w = heatmap.shape
    cx, cy = int(center_xy[0]), int(center_xy[1])

    left, right = min(cx, radius), min(w - cx, radius + 1)
    top, bottom = min(cy, radius), min(h - cy, radius + 1)
    if left + right <= 0 or top + bottom <= 0:
        return

    yy, xx = np.ogrid[-top:bottom, -left:right]
    gaussian = np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma + 1e-9))
    gaussian[gaussian < np.finfo(gaussian.dtype).eps * gaussian.max()] = 0

    patch = heatmap[cy - top:cy + bottom, cx - left:cx + right]
    np.maximum(patch, gaussian, out=patch)


def build_targets(boxes, out_h, out_w, stride, min_overlap=0.7):
    """boxes: list of [x1,y1,x2,y2] in INPUT pixel coords.
    Returns heatmap (1,out_h,out_w), wh (2,out_h,out_w), offset (2,out_h,out_w),
    mask (out_h,out_w) float32 arrays, all at output (stride-reduced) resolution."""
    heatmap = np.zeros((1, out_h, out_w), dtype=np.float32)
    wh = np.zeros((2, out_h, out_w), dtype=np.float32)
    offset = np.zeros((2, out_h, out_w), dtype=np.float32)
    mask = np.zeros((out_h, out_w), dtype=np.float32)

    for x1, y1, x2, y2 in boxes:
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            continue
        cx, cy = (x1 + x2) / 2.0 / stride, (y1 + y2) / 2.0 / stride
        cx_i, cy_i = int(cx), int(cy)
        if not (0 <= cx_i < out_w and 0 <= cy_i < out_h):
            continue
        radius = gaussian_radius(bw / stride, bh / stride, min_overlap)
        draw_gaussian(heatmap[0], (cx, cy), radius)
        wh[0, cy_i, cx_i] = bw
        wh[1, cy_i, cx_i] = bh
        offset[0, cy_i, cx_i] = cx - cx_i
        offset[1, cy_i, cx_i] = cy - cy_i
        mask[cy_i, cx_i] = 1.0

    return heatmap, wh, offset, mask


def decode_boxes(heatmap, wh, offset, stride, k=20, score_thresh=0.3):
    """heatmap: (1,H,W) sigmoid-activated torch tensor (single image, no batch dim).
    wh, offset: (2,H,W). Returns list of (x1,y1,x2,y2,score)."""
    hm = heatmap[0]
    pooled = torch.nn.functional.max_pool2d(hm[None, None], 3, stride=1, padding=1)[0, 0]
    peaks = (pooled == hm) & (hm > score_thresh)
    ys, xs = torch.nonzero(peaks, as_tuple=True)
    if len(xs) == 0:
        return []
    scores = hm[ys, xs]
    if len(xs) > k:
        top = torch.topk(scores, k).indices
        xs, ys, scores = xs[top], ys[top], scores[top]

    results = []
    for x, y, s in zip(xs.tolist(), ys.tolist(), scores.tolist()):
        ox, oy = offset[0, y, x].item(), offset[1, y, x].item()
        bw, bh = wh[0, y, x].item(), wh[1, y, x].item()
        cx, cy = (x + ox) * stride, (y + oy) * stride
        x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
        results.append((x1, y1, x2, y2, s))
    return results
