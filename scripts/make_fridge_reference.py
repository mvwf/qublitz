"""Write web/qublitz/tests/fridge-reference.json from the Python implementation.

The browser Fridge Noise Explorer (web/qublitz/fridge-engine.js) is a port of the
model in pages/Dilution_Refrigerator_Noise_Explorer.py. This script runs that
page's own helper functions on fixed inputs and records their outputs;
web/qublitz/tests/fridge-engine.test.js asserts the JS port matches.

Usage (from the repo root):
    python scripts/make_fridge_reference.py
"""
import ast
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(ROOT, "pages", "Dilution_Refrigerator_Noise_Explorer.py")
OUT = os.path.join(ROOT, "web", "qublitz", "tests", "fridge-reference.json")


def page_namespace():
    """Pull the constants and pure helpers out of the page without running its UI."""
    tree = ast.parse(open(PAGE, encoding="utf-8").read())
    keep = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and all(
                isinstance(t, ast.Name) and t.id in ("k_B", "h", "hbar") for t in node.targets):
            keep.append(node)
        elif isinstance(node, ast.FunctionDef):
            node.decorator_list = []          # drop @st.cache_data
            keep.append(node)
    ns = {"np": np}
    exec(compile(ast.Module(body=keep, type_ignores=[]), PAGE, "exec"), ns)
    return ns


def run_page(ns, temps, atten, p_in, cable, f_min, f_max, n_points):
    """Same steps as the page body, minus the Streamlit widgets."""
    freqs = np.linspace(f_min, f_max, n_points)
    n_eff, T_eff = ns["propagate_chain"](freqs, temps, atten)
    n_room = ns["thermal_n"](temps[0], freqs)
    red = 10 * np.log10(n_room / np.maximum(n_eff, 1e-30))
    P = p_in - np.cumsum(np.array(atten))
    P[-1] -= cable
    flux = np.array([ns["dBm_to_W"](p) / (ns["h"] * freqs * 1e9) for p in P])
    return freqs, n_eff, T_eff, red, P, flux


def finite(a):
    """JSON has no inf/nan; encode them as strings the test decodes."""
    return [float(x) if np.isfinite(x) else str(x) for x in np.asarray(a, dtype=float).ravel()]


def main():
    ns = page_namespace()
    default_T = [300.0, 50.0, 4.0, 0.7, 0.01]
    cases = [
        ("defaults", default_T, [0, 0, 20, 20, 20], -60, 15, 3, 6, 200),
        ("no attenuators", default_T, [0, 0, 0, 0, 0], -60, 15, 3, 6, 50),
        ("all 60 dB at 4 K", default_T, [0, 0, 60, 0, 0], -40, 0, 1, 20, 120),
        ("all 60 dB at MXC", default_T, [0, 0, 0, 0, 60], -20, 30, 4, 8, 60),
        ("warm MXC, attenuation everywhere", [300.0, 50.0, 4.0, 0.7, 0.1], [3, 10, 20, 10, 30], -80, 5.5, 2, 12, 77),
        ("stages at 0 K", [300.0, 50.0, 4.0, 0.0, 0.0], [0, 0, 20, 20, 20], -60, 15, 3, 6, 50),
        ("room at 0 K", [0.0, 50.0, 4.0, 0.7, 0.01], [0, 0, 20, 20, 20], -60, 15, 3, 6, 50),
    ]
    out = {"cases": [], "format_temp": []}
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for name, T, A, p_in, cable, f0, f1, n in cases:
            A = [float(a) for a in A]
            freqs, n_eff, T_eff, red, P, flux = run_page(ns, T, A, p_in, cable, f0, f1, n)
            out["cases"].append({
                "name": name, "temps": T, "atten": A, "p_in": p_in, "cable": cable,
                "f_min": f0, "f_max": f1, "n_points": n,
                "freqs": finite(freqs), "n_eff": [finite(r) for r in n_eff], "T_eff": [finite(r) for r in T_eff],
                "reduction_dB": [finite(r) for r in red], "P_dBm": finite(P), "flux": [finite(r) for r in flux],
            })
    for T in [0.0, 0.0099, 0.01, 0.7, 0.4999, 0.5, 4.0, 12.345, 50.0, 300.0]:
        out["format_temp"].append([T, ns["format_temp"](T)])
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"wrote {os.path.relpath(OUT, ROOT)} ({os.path.getsize(OUT) // 1024} KB)")


if __name__ == "__main__":
    main()
