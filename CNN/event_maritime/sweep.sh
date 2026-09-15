#!/usr/bin/env bash
# Hyperparameter sweep + augmentation ablations for the CNN event-maritime
# detector. Run from CNN/event_maritime with the event_maritime conda env
# active and a GPU available.
#
# Usage:
#   ./sweep.sh                  # full 150-epoch sweep (~6-7h on this GPU)
#   EPOCHS=60 ./sweep.sh        # faster coarse pass, then rerun the winner
#                                 with the default 150 epochs
#
# Each config gets its own sweep/<name>/{best.pt,last.pt} and sweep/<name>.log.
# Run summarize_sweep.py afterward (also run automatically at the end here)
# to rank configs by best val mean_best_iou.
set -e

EPOCHS=${EPOCHS:-150}
DATA_DIR=built_dvs
mkdir -p sweep

run() {
    name=$1; shift
    echo "=== $name ==="
    python3 train.py --data-dir "$DATA_DIR" --out-dir "sweep/$name" \
        --epochs "$EPOCHS" --batch-size 16 --device cuda --amp --workers 4 \
        "$@" 2>&1 | tee "sweep/$name.log"
}

# 1. baseline: identical to the current checkpoints/best.pt config, rerun here
#    so it's an apples-to-apples log alongside the rest.
run baseline        --width 96 --depth 8

# 2-4. one augmentation piece at a time, to see which actually help
run vflip           --width 96 --depth 8 --vflip
run rotate           --width 96 --depth 8 --rotate-deg 10
run photometric      --width 96 --depth 8 --gain-jitter 0.15 --gamma-jitter 0.15

# 5. all augmentations combined
run full_aug         --width 96 --depth 8 --vflip --rotate-deg 10 --gain-jitter 0.15 --gamma-jitter 0.15

# 6-8. hyperparameter variations on top of full augmentation
run full_aug_lr5e4   --width 96 --depth 8 --vflip --rotate-deg 10 --gain-jitter 0.15 --gamma-jitter 0.15 --lr 5e-4
run full_aug_wh02    --width 96 --depth 8 --vflip --rotate-deg 10 --gain-jitter 0.15 --gamma-jitter 0.15 --wh-weight 0.2
run full_aug_small   --width 64 --depth 6 --vflip --rotate-deg 10 --gain-jitter 0.15 --gamma-jitter 0.15

python3 summarize_sweep.py sweep
