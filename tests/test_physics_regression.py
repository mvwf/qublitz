"""UP-3's peer-review physics guard: before caching/gating any numpy compute
path, capture its output for fixed parameter sets and assert bit-equal output
post-refactor. A cache-key bug or an accidental refactor of the math itself
would otherwise silently serve WRONG PHYSICS to a student — the worst
possible failure mode for a teaching instrument, and one that a page-smoke
"did it render" test cannot catch (a wrong number renders fine).

These reference values were captured from the UNMODIFIED functions before
UP-3 wrapped their pages' inputs in st.form / added `st.fragment` boundaries
— none of that touches the math itself, so these must stay bit-identical
forever, not just today.
"""
import importlib.util
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _load(name, relpath):
    """Import a pages/*.py file as a plain module for its pure functions.

    These pages have no `if __name__ == "__main__"` guard, so their
    module-level Streamlit widget calls DO execute on import — but Streamlit
    tolerates being called with no active ScriptRunContext ("bare mode"):
    each widget just returns its own default value and prints a warning
    instead of raising. That's exactly what we want here; we only need the
    pure functions defined in the module, not a rendered page.
    """
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def laser():
    return _load("laser_heating_regression", "pages/Laser_Heating_Calculator.py")


@pytest.fixture(scope="module")
def dilution():
    return _load("dilution_regression", "pages/Dilution_Refrigerator_Noise_Explorer.py")


@pytest.fixture(scope="module")
def eptpd():
    return _load("eptpd_regression", "pages/EP_TPD_exploration.py")


@pytest.mark.parametrize("params,expected", [
    (([0.0, 1.0, 2.0], 4.0, 0.5, 1.0), [4.0, 4.316060279414279, 4.432332358381694]),
    (([0.0, 0.5, 1.0, 1.5], 0.02, 0.01, 0.3),
     [0.02, 0.028111243971624383, 0.029643260066527476, 0.029932620530009148]),
    (([0.0, 10.0, 20.0, 30.0], 4.0, 2.0, 15.0),
     [4.0, 4.9731657619348155, 5.472805723768547, 5.729329433526774]),
])
def test_temperature_trace_regression(laser, params, expected):
    result = laser.temperature_trace(*params)
    assert result == expected, "Laser_Heating_Calculator.temperature_trace() output changed"


@pytest.mark.parametrize("freqs,stage_temps,stage_atten_dB,expected_n_sum,expected_T_sum", [
    ([4.0, 5.0, 6.0], [300.0, 50.0, 4.0, 0.7, 0.01], [0.0, 0.0, 20.0, 20.0, 20.0],
     7802.962580294826, 1823.3684110184977),
    ([3.0, 4.5, 6.0], [300.0, 50.0, 4.0, 0.7, 0.01], [0.0, 0.0, 10.0, 10.0, 30.0],
     9589.022742163577, 1912.9453962671791),
    ([1.0, 5.0, 10.0], [300.0, 50.0, 4.0, 0.1, 0.02], [0.0, 5.0, 15.0, 25.0, 35.0],
     11835.617298567287, 1311.5533199742881),
])
def test_propagate_chain_regression(dilution, freqs, stage_temps, stage_atten_dB, expected_n_sum, expected_T_sum):
    n_eff, T_eff = dilution.propagate_chain(freqs, stage_temps, stage_atten_dB)
    # Tight relative tolerance, not exact `==`: summing a float array can
    # differ in the last bit or two depending on the exact reduction order
    # numpy/BLAS picks for a given run (confirmed empirically here — the
    # SAME inputs through the SAME function gave a 1-ULP-scale difference
    # between two otherwise-identical module-loading call sites). That's
    # floating-point noise, not a physics bug; 1e-9 relative is thousands of
    # orders of magnitude tighter than any real cache-key/math bug would be.
    assert float(n_eff.sum()) == pytest.approx(expected_n_sum, rel=1e-9), "propagate_chain() n_eff output changed"
    assert float(T_eff.sum()) == pytest.approx(expected_T_sum, rel=1e-9), "propagate_chain() T_eff output changed"


@pytest.mark.parametrize("phi,kappa,expected", [
    (0.0, 0.68, (174150.44271708722, 146951.6269752809, 229442.72372585122,
                 -7.79891706770286e-11, -144543459.49161988, 1.1641532182693481e-10)),
    (math.pi / 4, 1.2, (181897.705766976, 139874.95660660585, 145369.10255678312,
                        2.9103830456733704e-10, -187837110.34732386, -106066.01717798196)),
    (math.pi, 0.1, (10900.347790964546, 310201.72190140357, 615167.1992481393,
                    5.820766091346741e-11, -187745245.4781478, 1.1641532182693481e-10)),
])
def test_ep_tpd_fields_regression(eptpd, phi, kappa, expected):
    K_color, K_gray, instability, min_petermann, disc, tilde_q = eptpd._compute_ep_tpd_fields(phi, kappa)
    actual = (
        float(np.nansum(K_color)), float(np.nansum(K_gray)), float(instability.sum()),
        float(min_petermann.sum()), float(disc.sum()), float(tilde_q.sum()),
    )
    # Same reduction-order caveat as test_propagate_chain_regression above —
    # abs=1e-6 alongside rel=1e-9 because a couple of these fields
    # (min_petermann, tilde_q) sum to ~0 by construction (symmetric grid),
    # where a pure relative tolerance is meaningless.
    assert actual == pytest.approx(expected, rel=1e-9, abs=1e-6), "_compute_ep_tpd_fields() output changed"
