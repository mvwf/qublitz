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
