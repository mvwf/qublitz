"""Focused test for the QuBlitz Arena page (David's contribution).

Asserts the page renders without error and shows its title. The embed marker
assertion is tightened in C02-T4 once the live game is embedded.
"""
import os

from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARENA = os.path.join(ROOT, "pages", "QuBlitz_Arena.py")


def _run():
    at = AppTest.from_file(ARENA, default_timeout=60)
    at.run()
    return at


def test_arena_renders_without_exception():
    at = _run()
    assert not at.exception, f"Arena raised on load: {at.exception}"


def test_arena_shows_title():
    at = _run()
    titles = [t.value for t in at.title]
    assert any("QuBlitz Arena" in t for t in titles), f"title not found in {titles}"
