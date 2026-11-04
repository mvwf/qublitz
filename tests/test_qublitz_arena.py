"""Focused tests for the QuBlitz Arena page (David's contribution).

Asserts the page renders, shows its title, carries the academic framing
(the concept map / objectives), and that the live game is actually vendored
for the embed.
"""
import os

from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARENA = os.path.join(ROOT, "pages", "QuBlitz_Arena.py")
GAME_ASSET = os.path.join(ROOT, "pages", "_assets", "quantum_chess.html")


def _run():
    at = AppTest.from_file(ARENA, default_timeout=90)
    at.run()
    return at


def test_arena_renders_without_exception():
    at = _run()
    assert not at.exception, f"Arena raised on load: {at.exception}"


def test_arena_shows_title():
    at = _run()
    titles = [t.value for t in at.title]
    assert any("QuBlitz Arena" in t for t in titles), f"title not found in {titles}"


def test_arena_has_academic_framing():
    at = _run()
    blob = " ".join(m.value for m in at.markdown)
    assert "Born rule" in blob, "expected the concept map / Born-rule framing to render"
    assert "Learning objectives" in blob or "learning" in blob.lower()


def test_game_is_vendored_for_embed():
    # The embed reads this file; it must exist and be the real (large) game.
    assert os.path.exists(GAME_ASSET), "vendored quantum_chess.html missing"
    assert os.path.getsize(GAME_ASSET) > 100_000, "vendored game looks truncated"


# ── T1 bridge (QB-P0-6) ───────────────────────────────────────────────────────
#
# Added 2026-08-23 with the re-vendor. The game has always carried the receiver
# (importSimulatorParams reads window.QB_INITIAL_T1_NS), but nothing in this repo emitted the
# value, so the advertised simulator↔game link could not fire in the deployed product — and the
# vendored copy was the retired board build, which did not even contain the receiver. These
# tests pin both halves.

_GOOD_FIT = {"T1_ns": 250.0, "T1_ns_err": 3.0, "fit_curve": [], "ok": True}

# The phrase that appears only when the bridge actually fired.
_FIRED = "is driving the **Physics Lab** decoherence"


def _run_with_state(**state):
    at = AppTest.from_file(ARENA, default_timeout=90)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _messages(at):
    return [m.value for m in at.success] + [m.value for m in at.info]


def test_vendored_game_contains_the_t1_receiver():
    """
    The vendored copy must be the canonical build. Before the 2026-08-23 re-vendor it was the
    retired board build and contained none of these, so the bridge could not have fired even
    with a transmitter in place.
    """
    html = open(GAME_ASSET, encoding="utf-8").read()
    for marker in ("importSimulatorParams", "QB_INITIAL_T1_NS", "Pulse Forge"):
        assert marker in html, f"vendored game is missing {marker!r} — is it the retired build?"


def test_bridge_is_quiet_without_a_measured_t1():
    at = _run_with_state()
    assert not at.exception
    joined = " ".join(_messages(at))
    assert "default decoherence model" in joined
    assert _FIRED not in joined


def test_bridge_fires_on_a_converged_fit():
    at = _run_with_state(td_t1_fit=_GOOD_FIT)
    assert not at.exception
    joined = " ".join(_messages(at))
    assert "250.0 ns" in joined and _FIRED in joined


def test_unconverged_fit_is_not_a_measurement():
    """
    ok=False is _fit_t1_trace's initial guess (half the trace duration), returned when scipy's
    curve_fit is unavailable. A guess must not drive the game's physics.
    """
    at = _run_with_state(td_t1_fit={**_GOOD_FIT, "ok": False})
    assert not at.exception
    assert _FIRED not in " ".join(_messages(at))


def test_nonsense_fit_values_are_rejected():
    for bad in (0.0, -5.0, float("nan"), float("inf"), None, "250", True, [250]):
        at = _run_with_state(td_t1_fit={**_GOOD_FIT, "T1_ns": bad})
        assert not at.exception, f"raised on T1_ns={bad!r}"
        assert _FIRED not in " ".join(_messages(at)), f"accepted T1_ns={bad!r}"


def test_the_hidden_simulator_secret_never_reaches_the_game():
    """
    THE guard on this feature. user_data['T1'] is the true parameter Qubit_Simulator hides behind
    its 'show hidden simulator parameters' toggle because students are meant to measure it. The
    game displays the T1 it imports, so sourcing the bridge from that value would print the
    answer on the Arena page. Only a student's own fit may fire the bridge.
    """
    at = _run_with_state(user_data={"user": "student", "omega_q": 4.027,
                                    "omega_rabi": 0.35, "T1": 194.0})
    assert not at.exception
    joined = " ".join(_messages(at))
    assert "194" not in joined, "the hidden T1 leaked into the Arena page"
    assert _FIRED not in joined


def test_the_page_does_not_overclaim_what_the_bridge_controls():
    """
    Guards a real overclaim that shipped in the first draft of this feature and was caught by
    reading the game rather than trusting the docstring.

    importSimulatorParams() sets GAME.decoRate, and GAME.decoRate is read by
    drawDecoherenceDiagram() and the Physics Lab copy — and by nothing else. Duel evolution runs
    on DuelCore.K.T1_S/K.T2_S and never consults the import. So "your T1 is driving this game's
    decoherence" was false: it drives the Physics Lab model, not combat. On a lab platform used
    by students, that is an unearned physics claim, which is exactly what this project's
    no-vibes rule exists to stop.
    """
    at = _run_with_state(td_t1_fit=_GOOD_FIT)
    assert not at.exception
    joined = " ".join(_messages(at))
    assert "Physics Lab" in joined, "the message must scope the claim to the Physics Lab"
    assert "does **not** change duel combat" in joined, "the message must disclaim duel combat"


def test_release_copy_matches_the_laser_game():
    at = _run()
    text = " ".join(m.value for group in (at.markdown, at.caption, at.info, at.success) for m in group)
    for expected in ("Photon Duel", "Duel Training", "deprecated", "WASD", "Shift", "self-pulse", "Space"):
        assert expected in text
    for retired in ("Click the board", "**A** attack", "**G** guard", "heuristic Sage", "Claude Sage", "Move adjacent"):
        assert retired not in text, f"retired instruction still displayed: {retired}"


def test_untrusted_query_does_not_impersonate_a_fitted_measurement():
    at = AppTest.from_file(ARENA, default_timeout=90)
    at.query_params["t1"] = "194"
    at.run()
    assert not at.exception
    assert _FIRED not in " ".join(_messages(at))
