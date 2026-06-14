# Contributing to Qublitz

Thanks for contributing! This guide covers the local development setup and the
**verification gate** every change must pass before it is pushed or opened as a PR.

## Local setup

Qutip 4.7.6 has no wheel for Python 3.13, so use **Python 3.12**:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt    # app deps + pytest + ruff
```

On macOS, the qutip source build needs BLAS: `brew install openblas`.
On Debian/Ubuntu: `sudo apt-get install -y libopenblas-dev`.

Run the app with `streamlit run home.py`.

## The verification gate — run it GREEN before every push / PR

```bash
bash scripts/verify.sh
```

This runs exactly what CI runs:

1. **ruff** — correctness lint (pyflakes + syntax).
2. **pytest** — smoke tests: every page must load without raising
   (`tests/test_pages_smoke.py`), plus the QuBlitz Arena page test.
3. **node** — the game's physics regression test, once the game is vendored.

It prints a single `VERIFY: PASS ✓` / `FAIL ✗` and exits non-zero on any failure.
CI (`.github/workflows/ci.yml`) enforces the same checks on push and PR.

## Pull-request checklist

- `bash scripts/verify.sh` is green.
- Branch named for the feature (e.g. `feature/qublitz-arena`).
- Commit messages follow `scope: imperative description` (e.g.
  `perf: cache mesolve in quantum_simulator`).
- PR description says **what** changed, **why**, and **how to test**, with a
  screenshot for any UI change.
- New/changed functions have docstrings.
