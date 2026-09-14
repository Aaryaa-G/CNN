"""Standard CenterNet losses: penalty-reduced pixelwise focal loss for the
heatmap, masked L1 for width/height and sub-pixel offset."""
import torch


def focal_loss(pred, target, alpha=2.0, beta=4.0):
    # Cast to float32 before clamping: under AMP, pred arrives as float16,
    # which can't represent 1 - 1e-6 distinctly from 1.0, so the clamp's
    # upper bound silently becomes 1.0 and log(1 - pred) below produces -inf.
    pred = pred.float().clamp(1e-6, 1 - 1e-6)
    target = target.float()
    pos_mask = (target == 1).float()
    neg_mask = (target < 1).float()
    neg_weight = torch.pow(1 - target, beta)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_mask
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weight * neg_mask

    n_pos = pos_mask.sum()
    loss = -(pos_loss.sum() + neg_loss.sum())
    return loss / n_pos if n_pos > 0 else -neg_loss.sum()


def masked_l1(pred, target, mask):
    mask = mask.unsqueeze(1)  # (B,1,H,W) broadcasting over the 2 wh/offset channels
    n = mask.sum() * pred.shape[1]
    if n == 0:
        return pred.sum() * 0.0
    return (torch.abs(pred - target) * mask).sum() / n


def centernet_loss(pred_hm, pred_wh, pred_off, gt_hm, gt_wh, gt_off, mask,
                    wh_weight=0.1, off_weight=1.0):
    hm_loss = focal_loss(pred_hm, gt_hm)
    wh_loss = masked_l1(pred_wh, gt_wh, mask)
    off_loss = masked_l1(pred_off, gt_off, mask)
    total = hm_loss + wh_weight * wh_loss + off_weight * off_loss
    return total, {"hm": hm_loss.item(), "wh": wh_loss.item(), "off": off_loss.item()}
