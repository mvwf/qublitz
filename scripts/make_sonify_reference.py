"""Write web/qublitz/tests/sonify-reference.json from the Python implementation.

The browser Sonify page (web/qublitz/sonify-engine.js) is a port of the signal path
in pages/Sonify.py and utils/sound_to_image.py. This script runs those functions on
fixed inputs and records their outputs; web/qublitz/tests/sonify-engine.test.js
asserts the JS port matches.

Usage (from the repo root):
    python scripts/make_sonify_reference.py
"""
import ast
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from utils.sound_to_image import audio_to_image_array  # noqa: E402

PAGE = os.path.join(ROOT, "pages", "Sonify.py")
OUT = os.path.join(ROOT, "web", "qublitz", "tests", "sonify-reference.json")
AUDIO_STRIDE = 7     # keep every 7th audio sample
IMAGE_STRIDE = 5


def page_namespace():
    """Pull DEFAULT_SR and the helper functions out of the page without running its UI."""
    tree = ast.parse(open(PAGE, encoding="utf-8").read())
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name != "main")
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "DEFAULT_SR" for t in n.targets))]
    ns = {"np": np}
    exec(compile(ast.Module(body=keep, type_ignores=[]), PAGE, "exec"), ns)
    return ns


def main():
    ns = page_namespace()
    rng = np.random.default_rng(23)
    out = {"sr": ns["DEFAULT_SR"], "audio_stride": AUDIO_STRIDE, "image_stride": IMAGE_STRIDE, "rows": [], "images": []}
    # Image rows: smooth envelopes plus noise; power-of-two and other widths.
    for cols, duration, f in [(1024, 1.0, 65.41), (800, 0.5, 65.41), (777, 0.37, 440.0), (4096, 0.25, 110.0)]:
        x = np.linspace(0, 1, cols)
        env = np.clip(0.5 + 0.4 * np.sin(2 * np.pi * 3 * x) + 0.1 * rng.standard_normal(cols), 0, 1)
        t = ns["get_time_axis"](cols, 0.0, duration)
        mod = ns["generate_modulated_matrix"](env[None, :], t, np.array([f]))[0]
        mag, freqs = ns["compute_fft_map"](mod[None, :], duration / cols)
        audio, _ = ns["generate_audio"](env[None, :], t, 0, f, duration)
        out["rows"].append({"cols": cols, "duration": duration, "f": f, "env": env.tolist(), "t": t.tolist(),
                            "mod": mod.tolist(), "mag": mag[0].tolist(), "freqs": freqs.tolist(),
                            "audio_len": len(audio), "audio": audio[::AUDIO_STRIDE].astype(float).tolist()})
    # Sound -> image: short, block-averaged (with edge padding) and constant inputs. The input
    # is a formula the JS test rebuilds, so only every IMAGE_STRIDE-th output value is stored.
    for n in (3000, 20001, 300):
        i = np.arange(n)
        a = np.sin(i * 0.013) * np.exp(-i / n) + 0.05 * np.sin(i * i * 0.0007) if n != 300 else np.full(n, 0.3)
        img = audio_to_image_array(a)
        out["images"].append({"n": n, "shape": list(img.shape), "row": img[0][::IMAGE_STRIDE].tolist(),
                              "rows_equal": bool((img == img[0]).all())})
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
