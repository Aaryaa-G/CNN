"""Parse sweep.sh's per-config training logs and rank them by the best val
mean_best_iou each one reached.

Usage:
    python summarize_sweep.py sweep
"""
import glob
import os
import re
import sys

EPOCH_RE = re.compile(r"^Epoch (\d+):")
VAL_RE = re.compile(r"val: mean_best_iou=([\d.]+) precision=([\d.]+) recall=([\d.]+)")


def parse_log(path):
    best = None
    cur_epoch = None
    with open(path) as f:
        for line in f:
            m = EPOCH_RE.match(line)
            if m:
                cur_epoch = int(m.group(1))
                continue
            m = VAL_RE.search(line)
            if m:
                iou, prec, rec = map(float, m.groups())
                if best is None or iou > best[1]:
                    best = (cur_epoch, iou, prec, rec)
    return best


def main():
    log_dir = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    rows = []
    for path in sorted(glob.glob(os.path.join(log_dir, "*.log"))):
        name = os.path.splitext(os.path.basename(path))[0]
        result = parse_log(path)
        if result:
            epoch, iou, prec, rec = result
            rows.append((name, epoch, iou, prec, rec))

    if not rows:
        print(f"No logs with val lines found under {log_dir}/*.log")
        return

    rows.sort(key=lambda r: -r[2])
    print(f"{'config':<20}{'best_epoch':>11}{'mean_best_iou':>15}{'precision':>11}{'recall':>9}")
    for name, epoch, iou, prec, rec in rows:
        print(f"{name:<20}{epoch:>11}{iou:>15.4f}{prec:>11.4f}{rec:>9.4f}")


if __name__ == "__main__":
    main()
