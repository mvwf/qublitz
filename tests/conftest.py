"""Shared test setup for the Qublitz smoke suite.

Run everything from the repo root so that (a) pages loading relative assets
(``images/...``) resolve them, and (b) pages doing ``from quantum_simulator
import ...`` can find the top-level module.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
