"""
Simplified event-camera simulator.

No DVS-Voltmeter binary/library is available in this environment (it is a
separate MATLAB/Python research tool, not a pip package), so this
reimplements the standard *simplified* event-generation model that
DVS-Voltmeter, ESIM and v2e all build on: an event fires at a pixel whenever
its log-intensity changes by more than a contrast threshold since the last
frame, with polarity given by the sign of the change and the event count
given by how many multiples of the threshold were crossed. What we do NOT
reproduce is DVS-Voltmeter's per-pixel stochastic noise model (its Brownian
voltage-leak parameters k1..k6) -- this is a deterministic, noise-free
approximation, which is standard practice for synthetic training data when
the exact simulator is unavailable.

Because we control the simulation, we generate exactly one event-frame per
consecutive pair of original VISO frames (t-1 -> t). This sidesteps the
timing-alignment guesswork the earlier ship_045-only pipeline needed (it had
to assume how 320 original frames mapped onto 50 externally-provided event
bins): here, event-frame i is *exactly* the transition into original frame
i+1, so it aligns perfectly with that frame's ground-truth boxes.
"""
import numpy as np
from PIL import Image

EPS = 1.0  # avoids log(0); also sets the effective noise floor


def load_gray(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def simulate_events_between(gray_prev, gray_curr, c_on=0.15, c_off=0.15, max_events_per_pixel=4):
    """Returns (on_counts, off_counts), each shape (H, W) float32, counting how
    many contrast-threshold crossings each pixel had between the two frames."""
    diff = np.log(gray_curr + EPS) - np.log(gray_prev + EPS)
    on_counts = np.clip(np.floor(diff / c_on), 0, max_events_per_pixel)
    off_counts = np.clip(np.floor(-diff / c_off), 0, max_events_per_pixel)
    return on_counts.astype(np.float32), off_counts.astype(np.float32)


def normalize_channel(x):
    m = x.max()
    return x / m if m > 0 else x


def pad_to_multiple(arr, multiple=32):
    """Pad H,W (last two dims of a (C,H,W) array) up to the next multiple with zeros.
    Padding is bottom/right only, so pixel (0,0) and all original box coordinates
    are unaffected."""
    c, h, w = arr.shape
    h_pad = (-h) % multiple
    w_pad = (-w) % multiple
    if h_pad == 0 and w_pad == 0:
        return arr, (h, w)
    out = np.zeros((c, h + h_pad, w + w_pad), dtype=arr.dtype)
    out[:, :h, :w] = arr
    return out, (h, w)
