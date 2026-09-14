"""Greedy IoU matching between decoded predictions and ground-truth boxes,
for a simple mean-best-IoU / precision / recall validation metric."""


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_frame(pred_boxes, gt_boxes, iou_thresh=0.3):
    """pred_boxes: list of (x1,y1,x2,y2,score). gt_boxes: list of [x1,y1,x2,y2].
    Returns (tp, fp, fn, sum_best_iou_over_gt)."""
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0, 0.0

    preds = sorted(pred_boxes, key=lambda p: -p[4])
    matched_gt = [False] * len(gt_boxes)
    tp = 0
    best_iou_sum = 0.0
    gt_best = [0.0] * len(gt_boxes)

    for p in preds:
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gt_boxes):
            if matched_gt[j]:
                continue
            iou = box_iou(p[:4], g)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0:
            gt_best[best_j] = max(gt_best[best_j], best_iou)
        if best_j >= 0 and best_iou >= iou_thresh:
            matched_gt[best_j] = True
            tp += 1

    fp = len(preds) - tp
    fn = sum(1 for m in matched_gt if not m)
    best_iou_sum = sum(gt_best)
    return tp, fp, fn, best_iou_sum
