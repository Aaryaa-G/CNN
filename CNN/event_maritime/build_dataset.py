"""
Build the full event-based maritime detection dataset from every sequence
found in the local VISO ship MOT folder (currently ship/045 and ship/047 --
whatever sequences exist under --viso-root/mot/*/* with an img/ and
gt/gt.txt will be picked up automatically, so adding more VISO sequences
later needs no code change).

Event generation now uses the REAL DVS-Voltmeter simulator (the paper's
stochastic-process model, cloned from GitHub into --dvs-voltmeter-root),
not the simplified log-diff approximation this pipeline started with.
Pass --simulator simplified to fall back to the old approximation (see
event_sim.py) if you ever need to run without the DVS-Voltmeter repo present.

For each sequence:
  1. Read frames in order (grayscale), assign each one a timestamp in
     microseconds assuming --fps (default 10, matching VISO's documented
     capture rate) frames per second of the ORIGINAL satellite video.
  2. Feed frames one at a time to DVS-Voltmeter's EventSim. It returns, for
     every frame after the first, the actual asynchronous events
     (timestamp_us, x, y, polarity) generated between the previous frame and
     this one, using its full Brownian-motion-with-drift noise model (not
     just a fixed threshold crossing).
  3. Voxelize each transition's events into a 2-channel (OFF, ON) count
     "event frame", normalized to [0,1] per frame per channel -- same
     convention as the originally-provided ship_045_voxel_grid.npy. This is
     exactly one event-frame per original-frame transition, so it lines up
     1:1 with that frame's ground-truth boxes (no separate timestamp
     alignment step needed, unlike the first ship_045-only pipeline).
  4. Also write out the raw concatenated async event stream per sequence as
     a plain-text `timestamp x y polarity` file (same format the ship_045
     dataset originally shipped with) for reference / other uses.
  5. Pad each sequence's frames to a multiple of 32 (bottom/right zero-pad,
     coordinates unaffected) so the CNN's stride-4 downsampling divides
     evenly.
  6. Parse that sequence's gt.txt. Two conventions are present in this
     dataset and are auto-detected per file:
       - space-separated (ship/045 style): frame id x1 y1 x2 y2 ...
       - comma-separated (ship/047 style, standard MOTChallenge): frame,id,x,y,w,h,...

Output layout (under --out-dir, default ./built):
  built/
    ship_045_events.npy       (319, 2, Hpad, Wpad) float32  -- voxelized event frames
    ship_045_events_raw.txt              -- raw async (timestamp x y polarity) stream
    ship_047_events.npy       (299, 2, Hpad, Wpad) float32
    ship_047_events_raw.txt
    manifest.json
"""
import argparse
import json
import os
import glob

import cv2
import numpy as np

from event_sim import load_gray, simulate_events_between, normalize_channel, pad_to_multiple


def load_gt(path):
    """Returns dict: frame_id (1-indexed) -> list of [x1,y1,x2,y2]."""
    frames = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            comma = "," in line
            parts = line.split(",") if comma else line.split()
            frame_id = int(float(parts[0]))
            a, b, c, d = map(float, parts[2:6])
            if comma:
                # standard MOTChallenge: x, y, w, h
                x1, y1, x2, y2 = a, b, a + c, b + d
            else:
                # this dataset's ship/045-style: x1, y1, x2, y2 already
                x1, y1, x2, y2 = a, b, c, d
            frames.setdefault(frame_id, []).append([x1, y1, x2, y2])
    return frames


def find_sequences(viso_root):
    seqs = []
    for gt_path in glob.glob(os.path.join(viso_root, "mot", "*", "*", "gt", "gt.txt")):
        seq_dir = os.path.dirname(os.path.dirname(gt_path))
        category = os.path.basename(os.path.dirname(seq_dir))
        seq_name = os.path.basename(seq_dir)
        img_dir = os.path.join(seq_dir, "img")
        if os.path.isdir(img_dir):
            seqs.append({
                "name": f"{category}_{seq_name}",
                "img_dir": img_dir,
                "gt_path": gt_path,
            })
    return sorted(seqs, key=lambda s: s["name"])


def voxelize_events(events, h, w, max_events_per_pixel=None):
    """events: (N,4) int array [timestamp_us, x, y, polarity(0/1)]. Returns
    (off_counts, on_counts) each (H,W) float32."""
    off_counts = np.zeros((h, w), dtype=np.float32)
    on_counts = np.zeros((h, w), dtype=np.float32)
    if events is None or len(events) == 0:
        return off_counts, on_counts
    xs, ys, ps = events[:, 1], events[:, 2], events[:, 3]
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs, ys, ps = xs[valid], ys[valid], ps[valid]
    on_mask = ps == 1
    np.add.at(on_counts, (ys[on_mask], xs[on_mask]), 1.0)
    np.add.at(off_counts, (ys[~on_mask], xs[~on_mask]), 1.0)
    if max_events_per_pixel is not None:
        np.clip(on_counts, 0, max_events_per_pixel, out=on_counts)
        np.clip(off_counts, 0, max_events_per_pixel, out=off_counts)
    return off_counts, on_counts


def build_sequence_dvs_voltmeter(seq, out_dir, event_sim_cls, cfg, fps, save_raw=True):
    frame_paths = sorted(glob.glob(os.path.join(seq["img_dir"], "*")))
    n_frames = len(frame_paths)
    if n_frames < 2:
        raise RuntimeError(f"Sequence {seq['name']} has fewer than 2 frames, cannot simulate events")

    gt = load_gt(seq["gt_path"])

    first = cv2.imread(frame_paths[0], cv2.IMREAD_GRAYSCALE)
    native_h, native_w = first.shape
    dummy, _ = pad_to_multiple(np.zeros((2, native_h, native_w), dtype=np.float32))
    _, h, w = dummy.shape
    n_samples = n_frames - 1

    out_path = os.path.join(out_dir, f"{seq['name']}_events.npy")
    grid = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32,
                                      shape=(n_samples, 2, h, w))

    dt_us = int(round(1_000_000 / fps))
    sim = event_sim_cls(cfg=cfg, output_folder=out_dir, video_name=seq["name"])

    raw_f = open(os.path.join(out_dir, f"{seq['name']}_events_raw.txt"), "w") if save_raw else None

    samples = []
    t = 0
    for i, path in enumerate(frame_paths):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        events = sim.generate_events(img, t)  # None on the very first frame
        if events is not None:
            off_counts, on_counts = voxelize_events(events, native_h, native_w)
            frame_grid = np.stack([normalize_channel(off_counts), normalize_channel(on_counts)], axis=0)
            padded, _ = pad_to_multiple(frame_grid)
            grid[i - 1] = padded

            if raw_f is not None:
                np.savetxt(raw_f, events, fmt="%d")

            frame_id_1based = i + 1  # gt.txt is 1-indexed; this event frame is the
                                      # transition arriving at original frame i+1
            boxes = gt.get(frame_id_1based, [])
            samples.append({
                "sequence": seq["name"],
                "index_in_sequence": i - 1,
                "frame_id": frame_id_1based,
                "boxes": boxes,
            })
        t += dt_us

    if raw_f is not None:
        raw_f.close()
    grid.flush()
    del grid
    sim.reset()

    return {
        "name": seq["name"],
        "npy_path": os.path.basename(out_path),
        "native_h": native_h,
        "native_w": native_w,
        "padded_h": h,
        "padded_w": w,
        "n_samples": n_samples,
        "samples": samples,
    }


def build_sequence_simplified(seq, out_dir, c_on, c_off, max_events_per_pixel):
    """Old approximation (log-intensity threshold crossing, no real DVS
    noise model). Kept as a --simulator simplified fallback."""
    frame_paths = sorted(glob.glob(os.path.join(seq["img_dir"], "*")))
    n_frames = len(frame_paths)
    gt = load_gt(seq["gt_path"])

    first_gray = load_gray(frame_paths[0])
    native_h, native_w = first_gray.shape
    dummy, _ = pad_to_multiple(np.zeros((2, native_h, native_w), dtype=np.float32))
    _, h, w = dummy.shape
    n_samples = n_frames - 1

    out_path = os.path.join(out_dir, f"{seq['name']}_events.npy")
    grid = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32,
                                      shape=(n_samples, 2, h, w))

    samples = []
    prev_gray = first_gray
    for t in range(1, n_frames):
        curr_gray = load_gray(frame_paths[t])
        on_counts, off_counts = simulate_events_between(prev_gray, curr_gray, c_on, c_off, max_events_per_pixel)
        frame_grid = np.stack([normalize_channel(off_counts), normalize_channel(on_counts)], axis=0)
        padded, _ = pad_to_multiple(frame_grid)
        grid[t - 1] = padded

        frame_id_1based = t + 1
        boxes = gt.get(frame_id_1based, [])
        samples.append({
            "sequence": seq["name"],
            "index_in_sequence": t - 1,
            "frame_id": frame_id_1based,
            "boxes": boxes,
        })
        prev_gray = curr_gray

    grid.flush()
    del grid
    return {
        "name": seq["name"],
        "npy_path": os.path.basename(out_path),
        "native_h": native_h,
        "native_w": native_w,
        "padded_h": h,
        "padded_w": w,
        "n_samples": n_samples,
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--viso-root", default=os.path.join("Neuromorphic Camera", "VISO"),
                         help="Folder containing mot/<category>/<seq>/{img,gt}")
    parser.add_argument("--out-dir", default="built")
    parser.add_argument("--simulator", choices=["dvs_voltmeter", "simplified"], default="dvs_voltmeter")
    parser.add_argument("--dvs-voltmeter-root", default=os.path.join("..", "..", "DVS-Voltmeter-main", "DVS-Voltmeter-main"),
                         help="Folder containing DVS-Voltmeter's main.py and src/")
    parser.add_argument("--camera-type", default="DVS346", choices=["DVS346", "DVS240"])
    parser.add_argument("--fps", type=float, default=10.0,
                         help="Assumed original-video frame rate (VISO documents 10 fps); "
                              "sets the microsecond timestamp gap fed to DVS-Voltmeter")
    parser.add_argument("--no-raw", action="store_true", help="skip writing the raw event .txt files (saves disk/time)")
    # simplified-simulator-only options
    parser.add_argument("--c-on", type=float, default=0.15)
    parser.add_argument("--c-off", type=float, default=0.15)
    parser.add_argument("--max-events-per-pixel", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seqs = find_sequences(args.viso_root)
    if not seqs:
        raise RuntimeError(f"No sequences found under {args.viso_root}/mot/*/*/{{img,gt/gt.txt}}")

    print(f"Found {len(seqs)} sequence(s): {[s['name'] for s in seqs]}")
    print(f"Simulator: {args.simulator}")

    event_sim_cls, cfg = (None, None)
    if args.simulator == "dvs_voltmeter":
        from dvs_voltmeter_adapter import load_event_sim
        event_sim_cls, cfg = load_event_sim(args.dvs_voltmeter_root, args.camera_type)
        print(f"Loaded DVS-Voltmeter ({args.camera_type}, K={cfg.SENSOR.K}) from {args.dvs_voltmeter_root}")

    manifest = {"sequences": []}
    total_samples = 0
    total_boxes = 0
    for seq in seqs:
        print(f"Building {seq['name']} from {seq['img_dir']} ...")
        if args.simulator == "dvs_voltmeter":
            info = build_sequence_dvs_voltmeter(seq, args.out_dir, event_sim_cls, cfg, args.fps,
                                                 save_raw=not args.no_raw)
        else:
            info = build_sequence_simplified(seq, args.out_dir, args.c_on, args.c_off, args.max_events_per_pixel)
        n_boxes = sum(len(s["boxes"]) for s in info["samples"])
        print(f"  -> {info['n_samples']} event-frames, native {info['native_w']}x{info['native_h']}, "
              f"padded {info['padded_w']}x{info['padded_h']}, {n_boxes} ground-truth boxes")
        manifest["sequences"].append(info)
        total_samples += info["n_samples"]
        total_boxes += n_boxes

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    print(f"\nDone. {total_samples} total event-frames, {total_boxes} total ground-truth boxes.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
