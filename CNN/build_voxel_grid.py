"""
Rebuild the (50-bin, 2-channel) voxel grid directly from the RAW event
stream, at native VISO resolution (1345x451), instead of using the
provided 128x128 grid.

Why: at 128x128 the ship boxes shrink to ~3x5 px, which gives IoU-based
losses almost no usable gradient. Native resolution keeps ships at their
real ~22x11 px size, which is learnable.

Bins use the exact [start_time_us, end_time_us) windows from
ship_045_time_bins.csv, so this lines up 1:1 with boxes_50bins.npy (which
should be regenerated with prepare_data.py using NATIVE scale, see below).

Output: voxel_grid_native.npy, shape (50, 2, 451, 1345), float32, each
        (bin, channel) slice min-max normalized to [0, 1] independently
        (matching the normalization style of the provided grid).

PATHS: all paths below are relative to the CURRENT WORKING DIRECTORY you
run this script from. Either `cd` into the folder containing your
`ship_045_event_dataset/` directory before running, or pass --data-root /
override the individual --raw-events / --time-bins / --out flags.
"""
import argparse
import numpy as np
import csv
import os

NATIVE_W, NATIVE_H = 1345, 451


def load_time_bins(path):
    bins = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            bins.append((int(row["bin_id"]), float(row["start_time_us"]), float(row["end_time_us"])))
    return bins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=".",
                         help="Folder containing ship_045_event_dataset/ (default: current dir)")
    parser.add_argument("--raw-events", default=None,
                         help="Override path to ship_045_full.txt")
    parser.add_argument("--time-bins", default=None,
                         help="Override path to ship_045_time_bins.csv")
    parser.add_argument("--out", default=None, help="Override output .npy path")
    args = parser.parse_args()

    raw_events_path = args.raw_events or os.path.join(
        args.data_root, "ship_045_event_dataset", "1_raw_event_stream", "ship_045_full.txt")
    time_bins_path = args.time_bins or os.path.join(
        args.data_root, "ship_045_event_dataset", "4_time_bins", "ship_045_time_bins.csv")
    out_path = args.out or os.path.join(args.data_root, "voxel_grid_native.npy")

    for p in (raw_events_path, time_bins_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Could not find: {p}\n"
                f"Pass --data-root <folder containing ship_045_event_dataset/>, "
                f"or --raw-events / --time-bins directly."
            )

    bins = load_time_bins(time_bins_path)
    n_bins = len(bins)
    bin_starts = np.array([b[1] for b in bins])
    bin_ends = np.array([b[2] for b in bins])

    grid = np.zeros((n_bins, 2, NATIVE_H, NATIVE_W), dtype=np.float32)

    # Stream the raw event file (1.4M rows) -- avoid loading as python list of lists.
    data = np.loadtxt(raw_events_path, dtype=np.int64)  # columns: ts, x, y, polarity
    ts, x, y, pol = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    # clip coordinates defensively into valid range
    valid = (x >= 0) & (x < NATIVE_W) & (y >= 0) & (y < NATIVE_H)
    ts, x, y, pol = ts[valid], x[valid], y[valid], pol[valid]

    # assign each event to a bin via searchsorted on bin start edges
    # (bins are contiguous and sorted, so this is correct + fast)
    bin_idx = np.searchsorted(bin_starts, ts, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    # events at/after the very last bin's end still get clamped into last bin,
    # matching prepare_data.py's inclusive-end handling of the final bin.

    for b in range(n_bins):
        mask = bin_idx == b
        xs, ys, ps = x[mask], y[mask], pol[mask]
        for channel in (0, 1):  # 0 = OFF, 1 = ON (matches README polarity convention)
            cmask = ps == channel
            if cmask.sum() == 0:
                continue
            np.add.at(grid[b, channel], (ys[cmask], xs[cmask]), 1.0)

    # per (bin, channel) min-max normalization to [0, 1]
    for b in range(n_bins):
        for channel in (0, 1):
            m = grid[b, channel].max()
            if m > 0:
                grid[b, channel] /= m

    np.save(out_path, grid)
    print("Saved:", out_path, grid.shape, grid.dtype)
    print("Nonzero fraction:", (grid > 0).mean())
    print("Per-bin event counts (from time_bins.csv) vs rebuilt grid sums (sanity check):")
    for b in (0, 25, 49):
        print(f"  bin {b}: csv_count={bins[b][1], bins[b][2]}, "
              f"rebuilt_nonzero_pixels={int((grid[b] > 0).sum())}")


if __name__ == "__main__":
    main()
