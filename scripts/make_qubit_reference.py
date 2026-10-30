"""Write web/qublitz/tests/qubit-reference.json from the Python implementation.

The browser Qubit Simulator (web/qublitz/qubit-engine.js) is a port of
quantum_simulator.py and the fit/Ramsey helpers in pages/Qubit_Simulator.py.
This script runs those Python originals on fixed inputs and records their
outputs; web/qublitz/tests/qubit-engine.test.js asserts the JS port matches.

Each solve is recorded twice: with the solver settings the app uses
("default") and with much tighter tolerances ("tight"), so the test can tell
JS-vs-Python disagreement apart from qutip's own integration error.

Usage (from the repo root, in an environment with requirements-dev.txt):
    python scripts/make_qubit_reference.py
Works with qutip 4.7 (the pinned version) and qutip 5.
"""
import ast
import json
import os
import sys

import numpy as np
import qutip

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "web", "qublitz", "tests", "qubit-reference.json")
sys.path.insert(0, ROOT)

# --- qutip 5 compatibility: Options() was removed and mesolve's e_ops became keyword-only.
TIGHT = [False]
if not hasattr(qutip, "Options"):
    class _NoOptions:
        def __init__(self):
            raise RuntimeError("qutip 5: quantum_simulator.py falls back to an options dict")
    qutip.Options = _NoOptions
_mesolve = qutip.mesolve


def _mesolve_compat(H, rho0, tlist, c_ops=None, e_ops=None, args=None, options=None):
    if isinstance(options, dict):
        opts = dict(options)
        if TIGHT[0]:
            opts.update(atol=1e-12, rtol=1e-10, nsteps=10**7)
        return _mesolve(H, rho0, tlist, c_ops or [], e_ops=e_ops, args=args, options=opts)
    if TIGHT[0]:
        options.atol, options.rtol, options.nsteps = 1e-12, 1e-10, 10**7
    return _mesolve(H, rho0, tlist, c_ops, e_ops, args=args, options=options)


import quantum_simulator as qs  # noqa: E402

qs.mesolve = _mesolve_compat
solve = getattr(qs._solve_mesolve, "__wrapped__", qs._solve_mesolve)   # bypass st.cache_data


def page_functions(*names):
    """Pull pure helpers out of pages/Qubit_Simulator.py without running its UI."""
    path = os.path.join(ROOT, "pages", "Qubit_Simulator.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {"np": np}
    try:
        from scipy.optimize import curve_fit
        ns["curve_fit"] = curve_fit
    except ImportError:
        ns["curve_fit"] = None
    exec(compile(ast.Module(body=keep, type_ignores=[]), path, "exec"), ns)
    return ns


def square(n, t_final, start, stop, amp=1.0):
    t = np.linspace(0.0, t_final, n)
    env = np.zeros(n)
    env[(t >= start) & (t < stop)] = amp
    return env


def gaussian(n, t_final, center, sigma, amp=1.0):
    t = np.linspace(0.0, t_final, n)
    return amp * np.exp(-0.5 * ((t - center) / sigma) ** 2)


def _r(v):
    return float(f"{float(v):.10g}")


def solve_case(p):
    # ~400 evenly spaced output samples per case keeps the fixture small.
    idx = np.unique(np.linspace(0, p["n_steps"] - 1, min(p["n_steps"], 400)).astype(int))
    out = {"idx": idx.tolist()}
    for label, tight in (("default", False), ("tight", True)):
        TIGHT[0] = tight
        expect, probs = solve(p["omega_q"], p["omega_rabi"], p["t_final"], p["n_steps"], p["omega_d"],
                              np.asarray(p["env_I"], float), np.asarray(p["env_Q"], float), p["T1"], p["T2"])
        out[label] = {"x": [_r(expect[0][i]) for i in idx], "y": [_r(expect[1][i]) for i in idx],
                      "z": [_r(expect[2][i]) for i in idx], "p1": [_r(probs[i]) for i in idx]}
    TIGHT[0] = False
    return out


def main():
    cases = []

    def add(name, **p):
        p.setdefault("env_Q", np.zeros(p["n_steps"]))
        case = {"name": name, "params": {k: ([_r(x) for x in v] if isinstance(v, np.ndarray) else v)
                                         for k, v in p.items()}}
        case.update(solve_case(p))
        cases.append(case)

    N = 24   # tests/test_simulator.py's COMMON parameters
    add("resonant Rabi, no decay (test_simulator COMMON)", omega_q=5.0, omega_rabi=0.05, t_final=20.0,
        n_steps=N, omega_d=5.0, env_I=np.ones(N), T1=0.0, T2=0.0)
    n = 25 * 200
    add("square pi-ish pulse then T1 decay", omega_q=4.3, omega_rabi=0.02, t_final=200.0, n_steps=n,
        omega_d=4.3, env_I=square(n, 200.0, 5.0, 30.0), T1=150.0, T2=300.0)
    n = 25 * 120
    add("detuned Gaussian on Q with decay", omega_q=4.45, omega_rabi=0.03, t_final=120.0, n_steps=n,
        omega_d=4.46, env_I=np.zeros(n), env_Q=gaussian(n, 120.0, 40.0, 8.0, 0.8), T1=80.0, T2=160.0)
    n = 25 * 60
    add("I and Q pulses, negative amplitude", omega_q=4.1, omega_rabi=0.025, t_final=60.0, n_steps=n,
        omega_d=4.1, env_I=square(n, 60.0, 2.0, 12.0) - square(n, 60.0, 30.0, 40.0, 0.5),
        env_Q=square(n, 60.0, 15.0, 25.0), T1=200.0, T2=400.0)
    n = 25 * 25
    for wd in (4.2, 4.28, 4.3, 4.36):
        add(f"frequency-sweep column at {wd} GHz", omega_q=4.3, omega_rabi=0.02, t_final=25.0, n_steps=n,
            omega_d=wd, env_I=np.ones(n), T1=150.0, T2=300.0)

    # --- fits: run the page's own fitters on fixed noisy data (noise stored, so JS fits the same numbers)
    f = page_functions("_rabi_fit_model", "_t1_fit_model", "_fit_rabi_trace", "_fit_t1_trace",
                       "_ramsey_fit_model", "_fit_ramsey_trace", "_rot_x", "_idle_propagator",
                       "_ramsey_p1_trace")
    rng = np.random.default_rng(1234)
    fits = []

    t = np.linspace(0.0, 150.0, 400)
    y = 0.5 - 0.48 * np.exp(-t / 90.0) * np.cos(2 * np.pi * 0.022 * t) + rng.normal(0, 0.02, t.size)
    r = f["_fit_rabi_trace"](t, y)
    fits.append({"kind": "rabi", "t": t.tolist(), "y": [_r(v) for v in y],
                 "expect": {k: r[k] for k in ("A", "f_MHz", "f_MHz_err", "phi", "tau_ns", "tau_ns_err", "C")}})

    t = np.linspace(0.0, 400.0, 300)
    y = 0.03 + 0.9 * np.exp(-t / 140.0) + rng.normal(0, 0.02, t.size)
    r = f["_fit_t1_trace"](t, y)
    fits.append({"kind": "t1", "t": t.tolist(), "y": [_r(v) for v in y],
                 "expect": {k: r[k] for k in ("A", "T1_ns", "T1_ns_err", "C")}})

    delays = np.linspace(0.0, 600.0, 121)
    ramsey = {}
    for det in (-1.5, 0.0, 0.5, 3.0):
        ramsey[str(det)] = f["_ramsey_p1_trace"](delays, det, 150.0, 300.0).tolist()
    # A cut with ~2 fringes over the window: at 0.5 MHz the trace is a fraction of one
    # fringe, frequency and T2 become degenerate and scipy's own answer is arbitrary.
    y = np.asarray(ramsey["3.0"]) + rng.normal(0, 0.03, delays.size)
    r = f["_fit_ramsey_trace"](delays, y, freq_guess_MHz=3.0)
    fits.append({"kind": "ramsey", "t": delays.tolist(), "y": [_r(v) for v in y], "freq_guess_MHz": 3.0,
                 "expect": {k: r[k] for k in ("A", "T2_ns", "T2_ns_err", "freq_MHz", "freq_MHz_err", "phi", "C")}})

    data = {
        "generated_by": "scripts/make_qubit_reference.py",
        "qutip": qutip.__version__,
        "numpy": np.__version__,
        "cases": cases,
        "ramsey": {"delays": delays.tolist(), "T1": 150.0, "T2": 300.0, "p1_by_detuning_MHz": ramsey},
        "fits": fits,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(data, fh, separators=(",", ":"))
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB, {len(cases)} solve cases, qutip {qutip.__version__})")


if __name__ == "__main__":
    main()
