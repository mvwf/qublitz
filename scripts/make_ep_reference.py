"""Write web/qublitz/tests/ep-reference.json from the Python implementation.

The browser EP/TPD explorer (web/qublitz/ep-engine.js) is a port of
pages/EP_TPD_exploration.py and the utils it imports. This script runs those
functions on fixed inputs and records their outputs;
web/qublitz/tests/ep-engine.test.js asserts the JS port matches.

Usage (from the repo root):
    python scripts/make_ep_reference.py
"""
import ast
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from utils.pt_peaks_MODEL import eigenvalues, peak_location  # noqa: E402
from utils.tpd_locations_nd import ep_location, tpd_location  # noqa: E402

PAGE = os.path.join(ROOT, "pages", "EP_TPD_exploration.py")
OUT = os.path.join(ROOT, "web", "qublitz", "tests", "ep-reference.json")
STRIDE = 25          # fields: keep every 25th row and column of the 500x500 grid
SWEEP_STRIDE = 10    # splitting: keep every 10th of the 2000 points


def compute_fields():
    """The page's _compute_ep_tpd_fields, without its @st.cache_data decorator."""
    tree = ast.parse(open(PAGE, encoding="utf-8").read())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_compute_ep_tpd_fields")
    fn.decorator_list = []
    ns = {"np": np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), PAGE, "exec"), ns)
    return ns["_compute_ep_tpd_fields"]


def splitting(phi, kappa_tilde_c):
    """Same steps as the second figure in the page's main(), minus Streamlit and Plotly."""
    J, f_c = 1.0, 0.0
    tpds, eps = tpd_location(phi, kappa_tilde_c), ep_location(phi)
    primary_tpd = next(d for d in tpds if d.degeneracy_type.name == "PRIMARY_TPD")
    primary_ep = next(d for d in eps if d.degeneracy_type.name == "PRIMARY_EP")
    dk_ep, df_ep = primary_ep.Delta_tilde_kappa, primary_ep.Delta_tilde_f
    if phi == 0:
        x2 = np.linspace(dk_ep - 1, dk_ep + 2.5, 2000)
    elif phi == np.pi:
        x2 = np.linspace(df_ep - 1, df_ep + 2.5, 2000)
    else:
        x2 = np.linspace(dk_ep - 0.5, -0.18, 2000)
    nu_plus, nu_minus, nu_0, lambda_plus, lambda_minus = (np.full_like(x2, np.nan) for _ in range(5))
    candidates, previous = [], None
    for i, x in enumerate(x2):
        df, dk = (x, 0) if phi == np.pi else ((2 * np.sin(phi)) / x, x)
        tpd_result = peak_location(J, f_c, kappa_tilde_c, df, dk, phi)
        ep_result = -1 * eigenvalues(J, f_c, kappa_tilde_c, df, dk, phi).imag
        if len(tpd_result) == 2:
            nu_plus[i], nu_minus[i] = tpd_result
        else:
            nu_0[i] = tpd_result[0]
        lambda_plus[i], lambda_minus[i] = ep_result
        delta_lambda = np.sqrt(-df**2 + 2 * df * dk * 1j + dk**2 - 4 * J * np.exp(1j * phi))
        current = delta_lambda.real - (kappa_tilde_c - dk)
        if previous is not None and np.sign(current) != np.sign(previous):
            candidates.append(df if phi == np.pi else dk)
        previous = current
    target = df_ep if phi == np.pi else dk_ep
    instability = min(candidates, key=lambda v: abs(v - target)) if candidates else None
    yvals = np.concatenate([lambda_plus, lambda_minus, nu_plus, nu_minus, nu_0])
    s = slice(None, None, SWEEP_STRIDE)
    return {
        "x": x2[s], "nu_plus": nu_plus[s], "nu_minus": nu_minus[s], "nu_0": nu_0[s],
        "lambda_plus": lambda_plus[s], "lambda_minus": lambda_minus[s],
        "tpd": primary_tpd.Delta_tilde_f if phi == np.pi else primary_tpd.Delta_tilde_kappa,
        "ep": target, "instability": instability,
        "ymin": float(np.nanmin(yvals)), "ymax": float(np.nanmax(yvals)),
    }


def clean(v):
    """JSON-safe: arrays to lists, NaN/inf to strings."""
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [clean(x) for x in v]
    if v is None or isinstance(v, str):
        return v
    v = float(v)
    return v if np.isfinite(v) else ("nan" if np.isnan(v) else ("inf" if v > 0 else "-inf"))


def degeneracies(lst):
    return [[d.degeneracy_type.name, float(d.Delta_tilde_kappa), float(d.Delta_tilde_f)] for d in lst]


def main():
    fields_fn = compute_fields()
    phases = [k * np.pi / 8 for k in range(16)]
    out = {"stride": STRIDE, "sweep_stride": SWEEP_STRIDE, "fields": [], "locations": [], "splitting": []}
    for k, kc in [(0, 0.68), (2, 1.2), (8, 0.68), (11, 2.5), (5, 0.0)]:
        K_color, K_gray, instability, min_pet, disc, tilde_q = fields_fn(phases[k], kc)
        K = np.where(np.isnan(K_color), K_gray, K_color)
        s = slice(None, None, STRIDE)
        out["fields"].append({"k": k, "kc": kc, "K": K[s, s], "instability": instability[s, s],
                              "min_petermann": np.broadcast_to(min_pet, K.shape)[s, s],
                              "disc": disc[s, s], "q": tilde_q[s, s]})
    for k, phi in enumerate(phases):
        for kc in (0.0, 0.68, 1.5, 2.0, 2.5):
            out["locations"].append({"k": k, "kc": kc, "ep": degeneracies(ep_location(phi)),
                                     "tpd": degeneracies(tpd_location(phi, kc))})
    for k, kc in [(0, 0.68), (0, 2.5), (8, 0.68), (8, 1.9), (2, 0.68), (4, 1.0), (11, 0.68), (14, 2.2)]:
        out["splitting"].append({"k": k, "kc": kc, **splitting(phases[k], kc)})
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(clean(out), f, separators=(",", ":"))
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
