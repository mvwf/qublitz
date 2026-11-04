"""Photon Duel and Duel Training, embedded as the platform's QuBlitz Arena.

The retired board game remains in Git history. The fitted-T1 bridge affects
Physics Lab diagnostics only; duel combat uses its own fixed time constants.
"""
import json
import math
from pathlib import Path
from typing import Optional

import streamlit as st

from utils.branding import load_logo
from utils.ui import page_header

_GAME_HTML = Path(__file__).parent / "_assets" / "quantum_chess.html"
_EMBED_HEIGHT = 920


@st.cache_data(show_spinner=False)
def _load_game_html() -> str:
    return _GAME_HTML.read_text(encoding="utf-8")


def _measured_t1_ns() -> Optional[float]:
    """Return only a converged student-fit T1, never the hidden simulator parameter.

    The embedded game's receiver applies this to Physics Lab diagnostics, not
    duel combat. Reject the solver's unconverged initial guess and malformed
    values so the page cannot present either as a measured result.
    """
    fit = st.session_state.get("td_t1_fit")
    if not isinstance(fit, dict) or not fit.get("ok"):
        return None
    raw = fit.get("T1_ns")
    # Numbers only, and no bools (bool is an int subclass). _fit_t1_trace always writes a float,
    # so a string here means something other than the simulator wrote this key — and quietly
    # coercing unidentified input into the parameter that drives the game's physics is not a
    # robustness win, it is an unowned number.
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    t1 = float(raw)
    if not math.isfinite(t1) or t1 <= 0:
        return None
    return t1


def _render_sidebar():
    st.sidebar.image(load_logo("images/qublitz.png"))
    st.sidebar.image(load_logo("images/logo.png"))
    st.sidebar.markdown(
        '<div style="text-align:center;"><a href="https://sites.google.com/view/fitzlab/home" '
        'target="_blank" style="font-size:1.2rem; font-weight:bold;">FitzLab Website</a></div>',
        unsafe_allow_html=True,
    )


def _render_embed():
    inject = ""

    # srcdoc cannot read the parent query string; relay only the student's fitted value.
    t1_ns = _measured_t1_ns()
    if t1_ns is not None:
        inject += f"<script>window.QB_INITIAL_T1_NS={json.dumps(t1_ns)};</script>"

    st.components.v1.html(inject + _load_game_html(), height=_EMBED_HEIGHT, scrolling=True)

    if t1_ns is not None:
        st.success(
            f"Your measured **T₁ = {t1_ns:.1f} ns** is driving the **Physics Lab** decoherence "
            "model in the game above — open Physics Lab to see the relaxation curves redrawn "
            "against your own value, scaled to a 200 ns reference so a qubit with twice the "
            "reference T₁ decays twice as slowly there.\n\n"
            "It does **not** change duel combat: the duel evolves on its own fixed T₁/T₂ "
            "(90 s / 30 s in open space), and whether an imported T₁ should set duel balance is "
            "an open design question, not something this page quietly decides.",
            icon=":material/link:",
        )
    else:
        st.info(
            "The Physics Lab in the game above is showing its default decoherence model. Run a "
            "time-domain simulation on **Qubit Simulator** and fit T₁ from the relaxation tail — "
            "your fitted value will then drive those curves. The bridge uses the T₁ *you "
            "measure*, never the simulator's hidden parameter.",
            icon=":material/science:",
        )

def _render_academic_framing():
    st.markdown(
        "**Photon Duel** is a real-time laser duel against a bot: each ship carries a qubit, "
        "laser pulses rotate its Bloch vector, and measurement resolves through the **Born rule**. "
        "**Duel Training** teaches the same mechanics through guided missions. Building quantum "
        "intuition is a learning goal; durable learning gains have not been established."
    )
    with st.expander(":material/track_changes: Learning objectives", expanded=True):
        st.markdown(
            "- **State preparation and phase** — choose X, Y, or T beams and hold a pulse to set its rotation.\n"
            "- **Measurement** — read P(|1⟩) before committing to a measurement shot.\n"
            "- **Decoherence** — observe T₁ relaxation and T₂ dephasing during real-time play.\n"
            "- **Entanglement** — explore correlated collapse through the duel's Bell link.\n"
            "- **Physics Lab** — inspect the state and compare diagnostic relaxation curves."
        )
    with st.expander(":material/schema: Concept map"):
        st.markdown(
            "| Action | Quantum concept |\n|---|---|\n"
            "| X / Y / T laser pulse | Rotation of the qubit's Bloch vector |\n"
            "| Self-pulse | Apply the selected rotation to your own qubit |\n"
            "| Measurement shot | Born-rule collapse; a target measured in \\|1⟩ loses |\n"
            "| Bell link | Entanglement and correlated measurement |\n"
            "| Time passing | Amplitude damping and dephasing |"
        )
    st.caption(
        "Built by David Mukuruva in the Fitzpatrick Lab, Dartmouth College. "
        "The game has physics regression tests; the simulator remains available through the links below."
    )


def _render_quickstart():
    st.markdown("#### How to play")
    st.markdown(
        "1. Choose **Duel Training** to learn, or **Photon Duel** for a match.\n"
        "2. Focus the game. **WASD** moves; **arrow keys** or the pointer aim. "
        "Hold and release **Shift** to fire, or **F** to self-pulse. "
        "Mouse left-hold/right-hold provide the same pulse actions.\n"
        "3. Select beams with **1 / 2 / 3**; **Space** measures; **E** entangles; **Esc** pauses. "
        "Read the live state before measuring."
    )
    st.caption(
        "Standard gamepad: left/right sticks move/aim, right/left triggers attack/self-pulse, "
        "LB/RB or D-pad select beams, A/Cross measures, X/Square entangles, Menu pauses. "
        "Keyboard and virtual standard-gamepad input are tested; physical controllers are not certified."
    )


def _render_related_links():
    # BR-4 — one link per learning objective that actually has a formalizing
    # page (relative phase and entanglement don't yet; not forcing a link that
    # isn't true). Mirrors the game's own in-engine TOME cross-links so the
    # bridge is the same both directions, not just Streamlit-side prose.
    st.markdown("**Keep exploring the platform**")
    col1, col2, col3 = st.columns(3)
    # st.page_link resolves relative to the app entrypoint (home.py). Guard it so
    # the page still renders if loaded in isolation (e.g. under AppTest, where the
    # sibling pages aren't on the entrypoint's page list).
    try:
        with col1:
            st.page_link("pages/Quantum_Measurement_Tutorial.py",
                         label="New to qubits? Start with the Measurement Tutorial",
                         icon=":material/school:")
        with col2:
            st.page_link("pages/Qubit_Simulator.py",
                         label="See the real physics in the Qubit Simulator",
                         icon=":material/science:")
        with col3:
            st.page_link("pages/Dilution_Refrigerator_Noise_Explorer.py",
                         label="Where T₁/T₂ decoherence actually comes from",
                         icon=":material/ac_unit:")
    except Exception:
        with col1:
            st.markdown("**New to qubits?** Open the *Quantum Measurement Tutorial* page.")
        with col2:
            st.markdown("**Want the real physics?** Open the *Qubit Simulator* page.")
        with col3:
            st.markdown("**Where does T₁/T₂ come from?** Open the *Dilution Refrigerator "
                         "Noise Explorer* page.")


def main():
    page_header(
        "QuBlitz Arena", "⚔️",
        "Photon Duel and Duel Training — laser pulses, qubit rotations, and Born-rule measurement.",
        "Play",
    )
    _render_sidebar()
    st.caption(
        "The former board game is deprecated and replaced here by Photon Duel and Duel Training. "
        "Historical versions remain in Git history; the Arena URL is unchanged."
    )
    st.caption(
        "Focus the game: WASD move · arrow keys aim · Shift attack · F self-pulse · "
        "1/2/3 beams · Space measure · E entangle · Esc pause."
    )
    _render_embed()
    st.divider()
    _render_quickstart()
    _render_academic_framing()
    st.divider()
    _render_related_links()


if __name__ == "__main__":
    main()
