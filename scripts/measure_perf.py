#!/usr/bin/env python3
"""UP-4 — measure first: cold-start / time-to-interactive / per-widget-change
recompute time on the four pages UP-3 targets, run BEFORE and AFTER its
gating changes so the PR bodies can cite real numbers, not assertions.
See docs/PERF_METHODOLOGY.md for how this works and how to re-run it.

Run: python3 scripts/measure_perf.py
"""
import os
import sys
import time

# `python3 scripts/measure_perf.py` puts scripts/ on sys.path[0], not the
# repo root — the pages import `utils.*` relative to the root, so add it.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit.testing.v1 import AppTest


def measure(name, path, widget_fn):
    t0 = time.perf_counter()
    at = AppTest.from_file(path, default_timeout=90)
    at.run()
    tti = time.perf_counter() - t0
    if at.exception:
        return name, tti, None, f"EXCEPTION on load: {at.exception}"

    t1 = time.perf_counter()
    try:
        widget_fn(at)
    except Exception as e:
        return name, tti, None, f"EXCEPTION driving widget: {e}"
    recompute = time.perf_counter() - t1
    status = "OK" if not at.exception else f"EXCEPTION after widget change: {at.exception}"
    return name, tti, recompute, status


def _dilution(at):
    # One of the stage-temperature number_inputs — actually feeds the
    # thermal_n()/noise-pipeline recompute, unlike toggling a display-only
    # checkbox (tried first; measured 0.06s, which undersold the real cost
    # since it doesn't drive the compute path this task is actually about).
    at.sidebar.number_input[0].increment()
    at.run()


def _laser(at):
    at.sidebar.number_input[0].increment()  # trigger_freq_mhz
    at.run()


def _tutorial(at):
    at.slider[0].set_value(0.5)  # exp = <sigma_z>, range -1..1, default 1.0
    at.run()


def _eptpd(at):
    at.slider[0].set_value(1.2)  # kappa_tilde_c, range 0..2.5, default 0.68
    at.run()


PAGES = [
    ("Dilution_Refrigerator_Noise_Explorer", "pages/Dilution_Refrigerator_Noise_Explorer.py", _dilution),
    ("Laser_Heating_Calculator", "pages/Laser_Heating_Calculator.py", _laser),
    ("Quantum_Measurement_Tutorial", "pages/Quantum_Measurement_Tutorial.py", _tutorial),
    ("EP_TPD_exploration", "pages/EP_TPD_exploration.py", _eptpd),
]

if __name__ == "__main__":
    print(f"{'page':<40} {'time-to-interactive':>20}  {'recompute-after-widget-change':>29}  status")
    for name, path, fn in PAGES:
        n, tti, recompute, status = measure(name, path, fn)
        rc = f"{recompute:.2f}s" if recompute is not None else "n/a"
        print(f"{n:<40} {tti:>19.2f}s  {rc:>29}  {status}")
