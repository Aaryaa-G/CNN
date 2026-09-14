# Ship/045 Event-Based Multi-Object Box Regression — Pipeline

## What this does
Trains a small CNN to regress bounding boxes for all **3 tracked ships**
in the VISO Ship/045 sequence, from event-camera voxel-grid input (50
temporal bins), using DVS-Voltmeter-synthesized events + VISO's MOT ground
truth.

## Files
- `build_voxel_grid.py` — rebuilds the voxel grid **at native VISO
  resolution (451x1345)** directly from `1_raw_event_stream/ship_045_full.txt`.
- `prepare_data.py` — aligns VISO's 320-frame MOT `gt.txt` (3 object IDs) to
  the 50 event time bins (via `4_time_bins/ship_045_time_bins.csv`),
  producing `boxes_50bins.npy`.
- `box_utils.py` — IoU / GIoU functions.
- `dataset.py` — PyTorch `Dataset`, with flip/shift/dropout augmentation.
- `model.py` — compact CNN (`ShipBoxNet`), see parameterization note below.
- `train.py` — K-fold cross-validated training + evaluation.

## Run order
```bash
python3 build_voxel_grid.py     # -> voxel_grid_native.npy (~240MB)
python3 prepare_data.py         # -> boxes_50bins.npy
python3 train.py --epochs 80 --k 5
```

## Critical findings / decisions made along the way

1. **The provided 128x128 voxel grid is unusable for box regression.**
   Ships are ~22x11 px in the native 1345x451 frame. Downsampled to
   128x128 that becomes ~3x5 px — I confirmed empirically (40 real training
   epochs) that IoU-based losses get essentially zero gradient signal at
   that scale (mean IoU stayed at 0.0000–0.0014 no matter how long trained).
   **Fix:** `build_voxel_grid.py` rebuilds the grid at native resolution
   directly from the raw event stream, so ships keep their real ~1.6-2.4%
   of the image size instead of being crushed to a few pixels.

2. **`gt.txt` is MOT format with 3 object IDs, not single-object.** The
   event dataset's README calls this "Single Object Tracking," but the VISO
   ground truth has 3 tracked ships per frame. Per your instruction, the
   model is trained as **multi-target** (3 boxes per bin), not single-object.

3. **Frame-rate mismatch, resolved by using event-stream-implied timing.**
   `ship_045.mp4` reports 10 fps / 320 frames / 32.0s, but the actual
   synthesized event stream spans only 10.627s. DVS-Voltmeter clearly used
   its own internal frame-interval assumption, not the video's real
   playback rate. `prepare_data.py` assumes the 320 original frames are
   evenly spaced across the **actual event-stream duration** (5766µs →
   10,633,225µs) — this is the only assumption consistent with the data
   you have. If you know the exact fps you passed to DVS-Voltmeter, that
   would let you build a more precise mapping; the `linspace` call in
   `prepare_data.py` is where to change it.

4. **Box aggregation per bin:** each of the 50 bins covers ~6.4 original
   frames on average (min 6, max 7); the label for a bin is the **average**
   of that object's box across all frames whose timestamp falls in the
   bin's window (per your instruction).

5. **The model head had to be reparameterized — this was the main bug.**
   A naive head that sigmoids all 4 corner coordinates independently over
   the full [0,1] image range starts training with wildly oversized boxes
   (e.g. predicted `[0.33, 0.17, 0.71, 0.37]` against a target of
   `[0.56, 0.28, 0.58, 0.30]` — a box ~15x too large) because nothing
   constrains predicted width/height to be small, and IoU gradients vanish
   once predicted/target boxes barely overlap. **Fix:** the head now
   predicts `(center_x, center_y, width, height)`, with width/height capped
   at `max_size_frac=0.15` of the frame, and **the output layer's bias is
   initialized from the training fold's actual mean box per object** (a
   "smart init" — the model literally starts by predicting the average
   ship box, and only has to learn small per-bin corrections). This alone
   took training-set IoU from ~0.001 (never improving) to ~0.45 after 60
   epochs on a 10-sample overfit test, and held-out cross-validated IoU
   from 0.0000 to ~0.13 after just 15 epochs / 2 folds in a smoke test —
   full training (80-100 epochs, 5-fold CV) should do meaningfully better.

## Known limitations to keep in mind
- **Only 50 labeled samples total.** K-fold CV (not a single split) is used
  so every bin gets validated once, but expect noisy, high-variance
  estimates of true performance. Heavier augmentation, or treating this as
  track-continuation (using previous-bin box as a strong prior/anchor
  instead of pure from-scratch CNN regression) may outperform a plain CNN
  at this sample size.
- **CPU training at native resolution is slow** (~1.6s per batch-of-8
  forward+backward on this machine's CPU). A GPU will make 80-100 epoch,
  5-fold runs practical; on CPU, expect each fold to take several minutes.
- **Frame-to-time mapping is an assumption** (see finding #3) — verify
  against DVS-Voltmeter's actual config if you have it.
