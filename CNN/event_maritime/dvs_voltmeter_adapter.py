"""Thin adapter around the real DVS-Voltmeter simulator (cloned from
https://github.com/Lynn0306/DVS-Voltmeter into ../../DVS-Voltmeter-main/DVS-Voltmeter-main
relative to this file's grandparent folder). This replaces the earlier
simplified log-diff simulator (event_sim.py, kept for reference / --simulator
simplified fallback) with the genuine stochastic-process event model from
the paper.
"""
import os
import sys


def load_event_sim(dvs_root, camera_type="DVS346", k=None):
    dvs_root = os.path.abspath(dvs_root)
    if not os.path.isdir(os.path.join(dvs_root, "src")):
        raise FileNotFoundError(
            f"Could not find a 'src' folder under {dvs_root}. "
            f"Pass --dvs-voltmeter-root pointing at the DVS-Voltmeter repo folder "
            f"that directly contains main.py and src/."
        )
    if dvs_root not in sys.path:
        sys.path.insert(0, dvs_root)

    from src.config import cfg
    from src.simulator import EventSim

    cfg.SENSOR.CAMERA_TYPE = camera_type
    if k is not None:
        cfg.SENSOR.K = list(k)
    elif camera_type == "DVS346":
        cfg.SENSOR.K = [0.00018 * 29250, 20, 0.0001, 1e-7, 5e-9, 0.00001]
    elif camera_type == "DVS240":
        cfg.SENSOR.K = [0.000094 * 47065, 23, 0.0002, 1e-7, 5e-8, 0.00001]
    else:
        raise ValueError(f"Unknown camera_type {camera_type}, and no --k override given")

    return EventSim, cfg
