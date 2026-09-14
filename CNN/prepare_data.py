"""
Align VISO Ship/045 MOT ground truth (320 frames, 3 objects) to the
50 event-stream time bins produced by DVS-Voltmeter, and rescale boxes
from native resolution (1345x451) to the voxel-grid resolution (128x128).

Output: boxes_50bins.npy, shape (50, 3, 4), format [x1, y1, x2, y2] in
        NATIVE 1345x451 pixel coordinates (float32) -- matching
        voxel_grid_native.npy (see build_voxel_grid.py). Boxes are the
        AVERAGE of all original-frame boxes whose (assumed) timestamp
        falls inside that bin.

PATHS: relative to the CURRENT WORKING DIRECTORY by default. Run from the
folder containing both `ship_045_event_dataset/` and the extracted
`Neuromorphic Camera/` folder, or override with --data-root / the
individual path flags.
"""
import argparse
import numpy as np
import csv
import os

NATIVE_W, NATIVE_H = 1345, 451
NUM_FRAMES = 320
NUM_OBJECTS = 3  # object ids 1, 2, 3

def load_gt(path):
    """Returns dict: frame_id (1-indexed) -> dict object_id -> [x1,y1,x2,y2]"""
    frames = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            frame_id, obj_id = int(float(parts[0])), int(float(parts[1]))
            x1, y1, x2, y2 = map(float, parts[2:6])
            frames.setdefault(frame_id, {})[obj_id] = [x1, y1, x2, y2]
    return frames

def load_time_bins(path):
    bins = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            bins.append((int(row["bin_id"]), float(row["start_time_us"]), float(row["end_time_us"])))
    return bins

def get_stream_span(path):
    ts_min, ts_max = None, None
    with open(path) as f:
        for line in f:
            t = int(line.split()[0])
            if ts_min is None or t < ts_min:
                ts_min = t
            if ts_max is None or t > ts_max:
                ts_max = t
    return ts_min, ts_max

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=".",
                         help="Folder containing ship_045_event_dataset/ and Neuromorphic Camera/ (default: current dir)")
    parser.add_argument("--gt", default=None, help="Override path to gt.txt")
    parser.add_argument("--time-bins", default=None, help="Override path to ship_045_time_bins.csv")
    parser.add_argument("--raw-events", default=None, help="Override path to ship_045_full.txt")
    parser.add_argument("--out", default=None, help="Override output .npy path")
    args = parser.parse_args()

    gt_path = args.gt or os.path.join(
        args.data_root, "Neuromorphic Camera", "VISO", "mot", "ship", "045", "gt", "gt.txt")
    time_bins_path = args.time_bins or os.path.join(
        args.data_root, "ship_045_event_dataset", "4_time_bins", "ship_045_time_bins.csv")
    raw_events_path = args.raw_events or os.path.join(
        args.data_root, "ship_045_event_dataset", "1_raw_event_stream", "ship_045_full.txt")
    out_path = args.out or os.path.join(args.data_root, "boxes_50bins.npy")

    for p in (gt_path, time_bins_path, raw_events_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Could not find: {p}\n"
                f"Pass --data-root <folder containing both extracted datasets>, "
                f"or --gt / --time-bins / --raw-events directly."
            )

    gt = load_gt(gt_path)
    bins = load_time_bins(time_bins_path)
    ts_min, ts_max = get_stream_span(raw_events_path)

    # Evenly space the 320 original frames across the actual event-stream duration.
    frame_times = np.linspace(ts_min, ts_max, NUM_FRAMES)  # frame i (0-indexed) -> time_us

    out = np.zeros((len(bins), NUM_OBJECTS, 4), dtype=np.float32)
    counts = np.zeros((len(bins),), dtype=np.int32)

    for bin_id, start_us, end_us in bins:
        # frames (1-indexed gt) whose assumed timestamp falls in [start, end)
        mask = (frame_times >= start_us) & (frame_times < end_us)
        frame_idxs_0based = np.nonzero(mask)[0]
        if bin_id == bins[-1][0]:  # last bin: include end boundary
            mask_last = (frame_times >= start_us) & (frame_times <= end_us)
            frame_idxs_0based = np.nonzero(mask_last)[0]

        if len(frame_idxs_0based) == 0:
            # no frame fell in this window (shouldn't normally happen with 320
            # frames / 50 bins ~ 6.4 frames/bin) -- fall back to nearest frame
            center = (start_us + end_us) / 2.0
            nearest = int(np.argmin(np.abs(frame_times - center)))
            frame_idxs_0based = np.array([nearest])

        per_obj_boxes = {oid: [] for oid in range(1, NUM_OBJECTS + 1)}
        for f0 in frame_idxs_0based:
            frame_id_1based = f0 + 1
            frame_boxes = gt.get(frame_id_1based, {})
            for oid in range(1, NUM_OBJECTS + 1):
                if oid in frame_boxes:
                    per_obj_boxes[oid].append(frame_boxes[oid])

        for oid in range(1, NUM_OBJECTS + 1):
            boxes = per_obj_boxes[oid]
            if len(boxes) == 0:
                # object missing for every frame in this window; hold last known
                out[bin_id, oid - 1] = out[bin_id - 1, oid - 1] if bin_id > 0 else 0.0
                continue
            arr = np.array(boxes, dtype=np.float64)
            avg = arr.mean(axis=0)  # [x1,y1,x2,y2] in native pixels -- no rescale
            out[bin_id, oid - 1] = avg

        counts[bin_id] = len(frame_idxs_0based)

    np.save(out_path, out)
    print("Saved:", out_path, out.shape)
    print("Frames-per-bin stats: min", counts.min(), "max", counts.max(), "mean", counts.mean())
    print("Sample bin 0 boxes (native 1345x451 coords):\n", out[0])
    print("Sample bin 25 boxes (native 1345x451 coords):\n", out[25])
    print("box x range:", out[..., [0, 2]].min(), out[..., [0, 2]].max())
    print("box y range:", out[..., [1, 3]].min(), out[..., [1, 3]].max())
    w = out[:, :, 2] - out[:, :, 0]
    h = out[:, :, 3] - out[:, :, 1]
    print(f"box width: min={w.min():.2f} max={w.max():.2f} mean={w.mean():.2f}")
    print(f"box height: min={h.min():.2f} max={h.max():.2f} mean={h.mean():.2f}")

if __name__ == "__main__":
    main()
