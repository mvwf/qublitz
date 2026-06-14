"""Smoke tests: every Streamlit page must load without raising.

Most pages are rendered headless with ``streamlit.testing.v1.AppTest`` and
asserted to raise no exception. Two pages cannot run headless in CI — they
either talk to a lab backend or auto-run a full ``mesolve`` on load — so they
are syntax/compile-checked instead. After C02-T12 removes the auto-simulation,
``Qubit_Simulator.py`` can graduate to the rendered list.
"""
import os
import py_compile

import pytest
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Pages safe to fully render headless (no external backend, no minutes-long solve).
RENDER_PAGES = [
    "home.py",
    "pages/QuBlitz_Arena.py",
    "pages/Laser_Heating_Calculator.py",
    "pages/Quantum_Measurement_Tutorial.py",
    "pages/IQ_mixer.py",
    "pages/Dilution_Refrigerator_Noise_Explorer.py",
    "pages/EP_TPD_exploration.py",
    "pages/Sonify.py",
]

# Pages that hit a lab API or auto-run a full mesolve on load: compile-check only.
COMPILE_ONLY = [
    "quantum_simulator.py",
    "pages/Custom_Qubit_Query.py",
    "pages/Qubit_Simulator.py",
]


@pytest.mark.parametrize("page", RENDER_PAGES)
def test_page_renders_without_exception(page):
    at = AppTest.from_file(os.path.join(ROOT, page), default_timeout=90)
    at.run()
    assert not at.exception, f"{page} raised on load: {at.exception}"


@pytest.mark.parametrize("page", COMPILE_ONLY)
def test_page_compiles(page):
    py_compile.compile(os.path.join(ROOT, page), doraise=True)
