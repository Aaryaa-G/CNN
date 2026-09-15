# Event-Based Maritime Ship Detector (VISO, full local dataset)

This extends the earlier ship/045-only, 50-bin experiment (see
`../README_pipeline.md`) to **every VISO ship sequence present locally**
(`ship/045`, 320 frames, 3 tracked ships; `ship/047`, 300 frames, 1 tracked
ship — these are the only sequences under `Neuromorphic Camera/VISO/mot/`;
there is no larger multi-category VISO data in this folder to add), converts
them to events using the **real DVS-Voltmeter simulator**, and trains a
proper multi-object detector instead of a fixed-3-box regressor, plus a
simple tracker to follow ships across frames.

## Why this pipeline differs from the ship_045-only one

1. **Real DVS-Voltmeter, not an approximation.** The actual simulator
   (github.com/Lynn0306/DVS-Voltmeter, the paper's stochastic
   Brownian-motion-with-drift model) is cloned into
   `../../DVS-Voltmeter-main/DVS-Voltmeter-main` and driven directly via
   `dvs_voltmeter_adapter.py` + `build_dataset.py`. `event_sim.py`'s
   simplified log-diff model is kept only as a `--simulator simplified`
   fallback for when the real repo isn't available.
2. **Frame-timing calibration matters a lot, and we matched it to your
   original dataset.** DVS-Voltmeter needs a timestamp (in microseconds) per
   frame; VISO's frames aren't individually timestamped, so we assume a
   constant frame rate (`--fps`). At `--fps 10` (VISO's nominally documented
   capture rate) the simulator produced **99%+ ON-polarity events** — almost
   no OFF events at all. Checking this against the *original* ship_045 event
   dataset you first gave me confirmed this isn't a bug in this pipeline:
   that dataset is **94% ON-biased too**. It's a genuine property of
   DVS-Voltmeter's default DVS346 parameters on slow-changing satellite
   footage — the constant dark-current drift terms (k4, k5 in the model)
   don't scale down with a larger inter-frame gap the way the
   signal-driven term does, so treating frames as far apart in time makes
   dark-current noise dominate. The original dataset's actual **event-stream
   duration (10.63s over 320 frames) implies ~30 fps**, not the video's
   nominal 10 fps — so `build_dataset.py` now defaults to `--fps 30` to
   match, which is less extreme (though still ON-biased — that's the model,
   not a bug) than the 10 fps run.
3. **One event-frame per original frame transition**, not 50 arbitrary
   bins. Event-frame *i* is exactly the transition into original frame
   *i+1*, so every sample aligns perfectly with that frame's ground truth —
   no timestamp-mapping assumption needed (this was flagged as an unverified
   guess in the old pipeline). This also gives ~618 training samples instead
   of 50.
4. **CenterNet-style detector, not fixed-N-box regression.** `ship/045` has
   3 ship IDs, `ship/047` has 1 — a fixed-output-size regressor (the old
   approach) can't generalize across sequences with different object
   counts. `model.py` predicts a class-agnostic ship heatmap + per-pixel
   width/height/offset, so it naturally handles however many ships are in a
   frame (0, 1, 2, 3...) and generalizes to any VISO ship sequence you add
   later. **Important:** the raw event frames only ever show ship *edges*
   (event cameras — real or simulated — only fire on brightness changes, so
   a ship's interior generates almost no events; this is true of genuine DVS
   footage too, not an artifact of simulation). The detector is trained
   against the *full* ground-truth box regardless, so a well-trained model
   should predict the whole ship extent from sparse edge evidence — that's
   the point of supervised training here, not something the input
   representation itself needs to show.
5. **Native resolution, per-sequence.** `ship/045` is 1345x451, `ship/047`
   is 512x512 — different sequences, kept at their own native size (padded
   only to the next multiple of 32 for the stride-4 CNN), batched by a
   `SequenceGroupedBatchSampler` so no cross-sequence padding is wasted.
6. **Two ground-truth conventions, auto-detected.** `ship/045/gt/gt.txt` is
   space-separated `frame id x1 y1 x2 y2 ...`; `ship/047/gt/gt.txt` is
   comma-separated standard MOTChallenge `frame,id,x,y,w,h,...`. Both were
   validated against image bounds and frame-to-frame box smoothness before
   writing `load_gt()` in `build_dataset.py` — if you add more VISO
   sequences, sanity-check this assumption against the new `gt.txt`.
7. **Detection only gives independent per-frame boxes — it doesn't "follow"
   anything by itself.** `track_sequence.py` adds a simple greedy IoU/
   centroid tracker on top of the trained detector to link boxes across
   frames into persistent, colored trails (drawn as a video) — this is what
   actually produces the "ship gets followed across the sequence" behavior.
   It's a basic greedy tracker, not a learned one (swap in SORT/ByteTrack if
   you need something more robust for real deployment).

## Files
- `event_sim.py` — the simplified fallback event simulator.
- `dvs_voltmeter_adapter.py` — loads the real DVS-Voltmeter `EventSim` /
  `cfg` from the cloned repo.
- `build_dataset.py` — discovers every sequence under
  `<viso-root>/mot/*/*/{img,gt/gt.txt}`, runs DVS-Voltmeter frame-by-frame,
  writes one memmapped `<seq>_events.npy` (N-1, 2, Hpad, Wpad) float32 per
  sequence, one `<seq>_events_raw.txt` (`timestamp x y polarity`, same
  format as your original dataset) per sequence, and `manifest.json`
  (per-sample box labels).
- `centernet_utils.py` — gaussian heatmap target encoding + box decoding.
- `dataset.py` — `EventManifestDataset` (chronological per-sequence
  train/val split, flip augmentation) + `SequenceGroupedBatchSampler`.
- `model.py` — `CenterNetLite`: stem (stride 4) + residual body + 3 heads
  (heatmap/wh/offset). `--width`/`--depth` scale it up for GPU training.
- `losses.py` — penalty-reduced focal loss (heatmap) + masked L1 (wh, offset).
- `eval_utils.py` — greedy IoU matching -> precision/recall/mean-best-IoU.
- `train.py` — the GPU training script.
- `infer_visualize.py` — draws predicted vs. ground-truth boxes on a few
  validation event-frames for a quick visual check.
- `track_sequence.py` — runs the detector across a whole sequence and links
  detections into trails, rendered as an `.mp4`.

## Run order

```bash
# 1. Build the dataset with the real DVS-Voltmeter simulator (run once;
#    ~10-13 min on CPU -- it's a physics simulation, much slower than a
#    simple frame-diff, but still doesn't need a GPU for this step)
python build_dataset.py --viso-root "../Neuromorphic Camera/VISO" --out-dir built_dvs --fps 30

# 2. Train on your GPU machine. --vflip/--rotate-deg/--gain-jitter/--gamma-jitter
#    were chosen by the sweep in sweep.sh (see RESULTS.md) -- rotation and
#    vertical flip alone each help substantially on this small dataset;
#    combined ("full_aug") is the best config found, +51% relative
#    mean_best_iou over no extra augmentation.
python train.py --data-dir built_dvs --out-dir checkpoints \
    --epochs 150 --batch-size 16 --device cuda --amp --workers 4 \
    --width 96 --depth 8 \
    --vflip --rotate-deg 10 --gain-jitter 0.15 --gamma-jitter 0.15

# 3. Visualize predictions on a handful of frames
python infer_visualize.py --data-dir built_dvs --checkpoint checkpoints/best.pt \
    --n 12 --out-dir preview --device cuda

# 4. Track a full sequence and render a trail video
python track_sequence.py --data-dir built_dvs --checkpoint checkpoints/best.pt \
    --sequence ship_045 --out preview/ship_045_tracked.mp4 --device cuda
```

On this dev machine (CPU only, no CUDA) a correctness smoke test was run
(`train.py --device cpu --max-train-batches N --epochs M`, small subsets) to
confirm the full forward/backward/decode/checkpoint loop works end to end —
see the conversation for the actual smoke-test numbers. **Full training was
not run here**; this machine has no GPU. `--device auto` (default) picks
CUDA automatically when you run it on a GPU machine.

## Known limitations
- **Still a small dataset by deep-learning standards** — 618 labeled
  samples from 2 sequences. It is the entire local VISO ship data;
  generalizing further would need more VISO sequences (other categories in
  the public VISO release — car/airplane/train — aren't ship-relevant and
  aren't included here) or additional maritime footage run through the same
  pipeline.
- **The `--fps` assumption is still a guess**, just a better-calibrated one
  (matched to the original dataset's actual event-stream duration rather
  than VISO's nominal 10 fps). If you know the real capture rate DVS-
  Voltmeter should assume, override `--fps`.
- **DVS346's default k-parameters give a heavy ON-event bias on this kind of
  footage** (see point 2 above) — expected/inherent to the model+content,
  not a bug, but worth knowing before interpreting the OFF channel (it will
  be sparse).
- **`ship/045` is padded from 1345x451 to 1376x480** (2.3%/6.4% margin) for
  stride-4 divisibility; `ship/047` needs no padding (512 already divides
  by 32).
- `track_sequence.py`'s tracker is a minimal greedy IoU/centroid matcher, not
  a learned or Kalman-filtered tracker — fine for a demo/sanity check, not
  for production-grade multi-object tracking under occlusion or fast motion.
