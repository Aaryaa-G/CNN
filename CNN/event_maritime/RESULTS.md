# CNN Baseline Results (Event-Based Maritime Ship Detection)

This is the finalized CNN component of the larger SNN / CNN / GNN /
Transformer comparative study. It documents the frozen evaluation protocol
and final numbers other architectures should be compared against.

## Task and data

- **Dataset**: VISO `ship/045` (320 frames, 3 tracked ships, native
  1345x451) and `ship/047` (300 frames, 1 tracked ship, native 512x512) --
  the only two ship sequences available locally.
- **Event representation**: 2-channel (OFF, ON) per-pixel event counts,
  simulated frame-to-frame with the real DVS-Voltmeter simulator (DVS346,
  `--fps 30`), normalized per frame, zero-padded to the next multiple of 32.
  See `build_dataset.py` and its module docstring for full details.
- **618 total event-frames** (319 for ship_045, 299 for ship_047), each
  aligned 1:1 with its originating frame's ground-truth boxes.
- **Split**: chronological 85/15 holdout *within each sequence* (last 15% of
  each sequence's frames are validation) -- 525 train / 93 val samples, 166
  ground-truth boxes in the val set. This split is fixed by
  `EventManifestDataset(val_frac=0.15)` and must be reused as-is for any
  architecture this CNN result is compared against.

## Model and training protocol

- **Model**: `CenterNetLite` (stride-4 stem + residual body + heatmap/wh/offset
  heads), `--width 96 --depth 8` (~1-2M params).
- **Loss**: penalty-reduced focal loss (heatmap) + masked L1 (width/height,
  weight 0.1; offset, weight 1.0).
- **Optimizer**: AdamW, lr 1e-3, weight decay 1e-4, cosine annealing over 150
  epochs, batch size 16, gradient-norm clipping at 10.0, mixed precision
  (AMP) on an NVIDIA RTX 4500 Ada.
- **Evaluation metric**: greedy IoU matching (`eval_utils.match_frame`,
  IoU threshold 0.3, detection score threshold 0.3) -> precision, recall,
  and mean-best-IoU (mean IoU of each ground-truth box's best-matching
  prediction, 0 if unmatched). This is the metric every architecture in the
  comparative study should report.

Two bugs were found and fixed in this codebase before these results were
produced (see git history): a float16-under-AMP clamp bug in the focal loss
that produced `nan`/`inf` losses and caused training to permanently diverge
partway through a run, and a missing gradient-clipping step. Both are fixed
in `losses.py` / `train.py`.

## Augmentation ablation

`sweep.sh` trained 8 configs, each for the full 150 epochs, to isolate the
effect of each augmentation and a few hyperparameter variants on top of the
best combination. Ranked by best val `mean_best_iou`:

| config | best epoch | mean_best_iou | precision | recall | notes |
|---|---|---|---|---|---|
| **full_aug** | 49 | **0.6713** | 0.7822 | 0.9518 | vflip + rotate(10deg) + gain/gamma jitter, all combined -- **adopted as the final baseline** |
| rotate (alone) | 93 | 0.6470 | 0.9371 | 0.8976 | single biggest lever on its own |
| full_aug_lr5e4 | 73 | 0.6100 | 0.8371 | 0.8976 | full_aug + lr 5e-4 -- lower LR didn't help |
| full_aug_wh02 | 141 | 0.5931 | 0.9267 | 0.8373 | full_aug + wh_weight 0.2 -- doubling the box-size loss weight didn't help |
| full_aug_small | 84 | 0.5855 | 0.8810 | 0.8916 | full_aug + width 64/depth 6 -- smaller model slightly underperforms |
| vflip (alone) | 25 | 0.5417 | 0.7234 | 0.8193 | second biggest single lever |
| baseline | 22 | 0.4453 | 0.8525 | 0.6265 | horizontal flip only (this pipeline's original setup) |
| photometric (alone) | 11 | 0.4263 | 0.6824 | 0.6988 | gain/gamma jitter alone *hurts* vs. baseline |

**Takeaway**: geometric augmentation (rotation, vertical flip) is what
actually helps on this small (525-sample), tiny-object dataset -- each adds
+0.10-0.20 mean_best_iou alone, and stacking them is roughly additive
(+0.226 combined). Photometric (gain/gamma) jitter alone is net-negative,
likely because it perturbs the exact signal (event magnitude) the model
reads to distinguish signal from DVS-Voltmeter's dark-current noise floor --
but it doesn't cancel out the geometric augmentations' gains when combined
with them in `full_aug`. Neither a lower learning rate, a heavier box-size
loss weight, nor a smaller model improved on `full_aug`.

## Final adopted baseline

- **Config**: `full_aug` (`--vflip --rotate-deg 10 --gain-jitter 0.15
  --gamma-jitter 0.15`, all other hyperparameters at their defaults).
- **Checkpoint**: `checkpoints/best.pt` (epoch 49/150).
- **Metrics** (val split, 93 samples / 166 boxes): mean_best_iou **0.6713**,
  precision **0.7822**, recall **0.9518**.
- **Reproduce**: see the "Run order" section of `README.md`, step 2.

## Known limitations (carried over from README.md, still apply)

- Small dataset by deep-learning standards -- 618 samples from 2 sequences,
  no more local VISO ship data exists to add.
- DVS-Voltmeter's DVS346 defaults give a heavy ON-polarity bias on this kind
  of slow-changing satellite footage (documented in README.md) -- expected
  behavior of the simulator on this content, not a bug.
- The `--fps 30` timestamp assumption is a calibrated guess, not a verified
  ground truth (see README.md point 2).
- `track_sequence.py`'s tracker is a minimal greedy IoU/centroid matcher, not
  learned or Kalman-filtered.
