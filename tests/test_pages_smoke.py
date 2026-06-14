"""Smoke tests: every Streamlit page must load without raising.

Every page is rendered headless with ``streamlit.testing.v1.AppTest`` and
asserted to raise no exception. Since C02-T12 gated the time-domain ``mesolve``
behind a Run button, the two simulator pages no longer auto-solve on load and
render headless too (the lab-API page short-circuits at its login prompt).
``quantum_simulator.py`` is a library, not a page, so it is compile-checked.
"""
import os
import py_compile

import pytest
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Pages safe to fully render headless (no external backend call, no auto-solve).
RENDER_PAGES = [
    "home.py",
    "pages/QuBlitz_Arena.py",
    "pages/Laser_Heating_Calculator.py",
    "pages/Quantum_Measurement_Tutorial.py",
    "pages/IQ_mixer.py",
    "pages/Dilution_Refrigerator_Noise_Explorer.py",
    "pages/EP_TPD_exploration.py",
    "pages/Sonify.py",
    "pages/Qubit_Simulator.py",
    "pages/Custom_Qubit_Query.py",
]

# Library modules (not Streamlit pages): syntax/compile-check only.
COMPILE_ONLY = [
    "quantum_simulator.py",
]


@pytest.mark.parametrize("page", RENDER_PAGES)
def test_page_renders_without_exception(page):
    at = AppTest.from_file(os.path.join(ROOT, page), default_timeout=90)
    at.run()
    assert not at.exception, f"{page} raised on load: {at.exception}"


@pytest.mark.parametrize("page", COMPILE_ONLY)
def test_page_compiles(page):
    py_compile.compile(os.path.join(ROOT, page), doraise=True)
