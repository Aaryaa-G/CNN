"""
Bounding-box utilities: format conversion + IoU/GIoU for training and eval.
All functions operate on tensors of shape (..., 4) in [x1, y1, x2, y2] format
unless noted otherwise.
"""
import torch


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Elementwise IoU between two sets of boxes with matching leading shape.
    boxes1, boxes2: (..., 4) in [x1,y1,x2,y2]. Returns (...,) IoU values.
    """
    x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
    y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
    x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
    y2 = torch.min(boxes1[..., 3], boxes2[..., 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h

    area1 = (boxes1[..., 2] - boxes1[..., 0]).clamp(min=0) * (boxes1[..., 3] - boxes1[..., 1]).clamp(min=0)
    area2 = (boxes2[..., 2] - boxes2[..., 0]).clamp(min=0) * (boxes2[..., 3] - boxes2[..., 1]).clamp(min=0)
    union = area1 + area2 - inter + eps
    return inter / union


def box_giou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Elementwise Generalized IoU. Returns (...,) GIoU values in [-1, 1]."""
    iou = box_iou(boxes1, boxes2, eps)

    x1 = torch.min(boxes1[..., 0], boxes2[..., 0])
    y1 = torch.min(boxes1[..., 1], boxes2[..., 1])
    x2 = torch.max(boxes1[..., 2], boxes2[..., 2])
    y2 = torch.max(boxes1[..., 3], boxes2[..., 3])
    enclosing_area = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0) + eps

    area1 = (boxes1[..., 2] - boxes1[..., 0]).clamp(min=0) * (boxes1[..., 3] - boxes1[..., 1]).clamp(min=0)
    area2 = (boxes2[..., 2] - boxes2[..., 0]).clamp(min=0) * (boxes2[..., 3] - boxes2[..., 1]).clamp(min=0)
    x1i = torch.max(boxes1[..., 0], boxes2[..., 0])
    y1i = torch.max(boxes1[..., 1], boxes2[..., 1])
    x2i = torch.min(boxes1[..., 2], boxes2[..., 2])
    y2i = torch.min(boxes1[..., 3], boxes2[..., 3])
    inter = (x2i - x1i).clamp(min=0) * (y2i - y1i).clamp(min=0)
    union = area1 + area2 - inter + eps

    giou = iou - (enclosing_area - union) / enclosing_area
    return giou


def giou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - GIoU, averaged over all boxes. pred/target: (..., 4)."""
    return (1.0 - box_giou(pred, target)).mean()
