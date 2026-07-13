"""QuBlitz Arena — David Mukuruva's contribution to the Fitzpatrick Lab platform.

Embeds the QuBlitz battle game (a self-contained quantum_chess.html) and frames
it as an academic learning companion: every game action maps to a real quantum
concept, and the in-engine physics is the same unit-tested open-quantum-system
model the simulator pages use. The game ties decoherence and relative phase to
real tactical stakes and motivates engagement with the concepts; the formal
teaching happens on the Physics Lab screen and the concept map/cross-links
below. Whether play transfers to durable understanding is a design goal this
project intends to measure later, not a claim this page asserts as proven.

Sage AI: the embedded game's Sage is a smart offline heuristic, permanently — no
key, no server-side proxy, nothing to configure. The live-LLM Sage path (a
server-side Anthropic proxy this docstring used to describe injecting via
``window.QB_SAGE_PROXY``) was removed from the game repo on 2026-07-08; this file
is the fork-side cleanup of that same removal (the JS side stopped reading that
global the same day — see the game repo's CLAUDE.md/HANDOFF.md for the full
rationale).
"""
from pathlib import Path

import streamlit as st

from utils.branding import load_logo

_GAME_HTML = Path(__file__).parent / "_assets" / "quantum_chess.html"
# Frame height tuned so the board + side panels + event log sit on one screen
# without inner scroll on a typical laptop; the game caps the board by viewport
# height (CSS min(...,100vh-…)) so it scales to fit this frame.
#
# Was 860 — an LLM-council review found the Sage panel/event log clipped at the
# bottom at that height. The game side responded by trimming vertical chrome
# and adding a Sage-panel collapse toggle, which should reduce (not eliminate)
# the pressure on this fixed frame. Bumped to 920 as a modest, judgment-call
# increase: a slightly taller default with a collapse option to fall back on
# beats a cramped fixed height with no give. This wasn't visually verified
# against the trimmed layout (no Streamlit run in this pass) — re-check on a
# typical laptop viewport and adjust if it still clips or now over-scrolls.
_EMBED_HEIGHT = 920


@st.cache_data(show_spinner=False)
def _load_game_html() -> str:
    return _GAME_HTML.read_text(encoding="utf-8")


def _render_sidebar():
    st.sidebar.image(load_logo("images/qublitz.png"))
    st.sidebar.image(load_logo("images/logo.png"))
    st.sidebar.markdown(
        '<div style="text-align:center;"><a href="https://sites.google.com/view/fitzlab/home" '
        'target="_blank" style="font-size:1.2rem; font-weight:bold;">FitzLab Website</a></div>',
        unsafe_allow_html=True,
    )


def _render_embed():
    # BR-1 — the game is injected via srcdoc, not a src= URL, so its own
    # location.search is never this page's query string; ?t1= has to be
    # relayed as an injected global, or quantum_chess.html's
    # importSimulatorParams() never sees it. st.query_params values are
    # always strings; validated numeric before injecting so a malformed URL
    # can't inject anything but a number into the page.
    inject = ""
    t1_param = st.query_params.get("t1")
    if t1_param is not None:
        try:
            t1_ns = float(t1_param)
            inject += f"<script>window.QB_INITIAL_T1_NS={t1_ns};</script>"
        except (TypeError, ValueError):
            pass
    st.components.v1.html(inject + _load_game_html(), height=_EMBED_HEIGHT, scrolling=True)
    st.info(
        "The offline heuristic Sage is active — concept-linked, per-unit advice with no API key "
        "and no third-party service anywhere in this project.",
        icon=":material/auto_awesome:",
    )


def _render_academic_framing():
    st.markdown(
        "**QuBlitz** is a quantum tactics game where every unit on the board is a **qubit**, "
        "every action is a **quantum gate**, and combat resolves via the **Born rule**. It is "
        "the interactive companion to this simulator: the same T₁/T₂ physics you explore on the "
        "other pages drive the game's decoherence — so decoherence and relative phase carry real "
        "tactical stakes here, not just flavor text. Building intuition for the math is the goal; "
        "the Physics Lab and concept map below do the formal teaching."
    )

    with st.expander(":material/track_changes: Learning objectives — what you'll reason about", expanded=True):
        st.markdown(
            "- **Superposition & state preparation** — drive a qubit between |0⟩, |1⟩, and |+⟩.\n"
            "- **The Born rule** — measurement outcomes are probabilistic in the qubit's amplitudes.\n"
            "- **Bloch-sphere state** — read a live qubit as a vector (the panel shows it each turn).\n"
            "- **T₁/T₂ decoherence** — watch charge relax exponentially when a unit sits idle.\n"
            "- **Relative phase** — discover why Z/S/T gates matter only through interference.\n"
            "- **Entanglement & the no-communication theorem** — Bell pairs that collapse together."
        )

    with st.expander(":material/schema: Concept map — every action maps to real quantum mechanics"):
        # Kets contain '|', the markdown table delimiter, so each is escaped as \| .
        st.markdown(
            "| In-game action | Quantum concept | In-engine rule |\n"
            "|---|---|---|\n"
            "| Charge with **X** / **H** | State preparation, superposition | X → \\|1⟩ (100%); H → \\|+⟩ (≈50%) |\n"
            "| **Attack** | Born rule | P(hit) = charge = P(\\|1⟩); a target in \\|1⟩ takes a CRITICAL |\n"
            "| **Idle** a turn | T₁/T₂ relaxation (Lindblad) | charge ∝ e^(−t/T₁), coherence ∝ e^(−t/T₂) |\n"
            "| **GUARD** | X-basis measurement, relative phase | crit risk = (1−x)/2; \\|+⟩ safe, \\|−⟩ exposed |\n"
            "| **Z / S / T** | relative-phase rotation | invisible to charge — matters only via the GUARD's X-basis |\n"
            "| **CNOT** | entanglement (Bell pair) | \\|Φ⁺⟩; measuring one collapses both; no signalling |\n"
            "| **MEASURE** | projective collapse | \\|ψ⟩ → \\|0⟩ or \\|1⟩, then stabilizes the unit |\n"
        )

    with st.expander(":material/science: The physics is real — and unit-tested"):
        st.markdown(
            "QuBlitz is not hand-waving flavor. Each unit evolves under the **exact Lindblad "
            "master-equation solution** for amplitude damping (T₁) plus dephasing (T₂), the same "
            "open-quantum-system model behind the simulator. Open the in-game **Physics Lab** to "
            "inspect a live density matrix ρ. The engine ships with a regression suite that asserts:\n\n"
            "- decoherence is **exponential** (R² > 0.99) and the fit recovers the input T₁ to within 1%,\n"
            "- a CNOT makes a true Bell pair with **perfectly correlated** collapse, and\n"
            "- a local gate on one half **never** changes the partner's marginal (no-communication)."
        )

    with st.expander(":material/school: Research context"):
        st.markdown(
            "Built by **David Mukuruva** within the **Fitzpatrick Lab, Dartmouth College** "
            "(PI: Prof. Mattias Fitzpatrick), as the game companion to the lab's qubit simulator. "
            "The roadmap calibrates the game's decoherence to **real measured T₁/T₂**, connecting "
            "the proposal's research questions — mapping BCTDS parameters to T₂ degradation, and "
            "characterizing non-Markovian (HEOM) memory effects — to an interactive learning tool."
        )

    with st.expander(":material/privacy_tip: Data & consent posture (GOV-2)"):
        st.markdown(
            "The short version: **no automatic network calls, ever.** Nothing is collected "
            "unless a student explicitly clicks the in-game **EXPORT MY GATE LOG** button, and "
            "even then the result is a local file download the student controls -- QuBlitz never "
            "transmits it anywhere itself. Full posture, exact export schema, and a FERPA note "
            "for classroom use: "
            "[`docs/DATA_AND_CONSENT.md`](https://github.com/dmukuruva-creator/Qublitz_Draft/blob/main/docs/DATA_AND_CONSENT.md) "
            "in the canonical game repo."
        )


def _render_quickstart():
    st.markdown("#### How to play in 30 seconds")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1 · Charge**")
        st.caption("Apply **X** (→ |1⟩, 100%) or **H** (→ |+⟩, 50%). A unit's charge *is* its P(|1⟩).")
    with c2:
        st.markdown("**2 · Close in**")
        st.caption("Move adjacent to a foe — but hurry: idle charge relaxes as e^(−t/T₁) every turn.")
    with c3:
        st.markdown("**3 · Collapse**")
        st.caption("**Attack** fires with probability = charge; a target caught in |1⟩ takes a CRITICAL.")


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
    st.set_page_config(page_title="QuBlitz Arena", layout="wide")
    _render_sidebar()

    st.title("QuBlitz Arena")
    st.caption(
        "Quantum mechanics with real stakes — every unit is a qubit, every move is a gate.  "
        "Click the board, then **A** attack · **H/X/Y/Z** charge · **M** measure · **C** entangle · "
        "**G** guard · **E** explain · **?** Sage."
    )

    # Game first, so it sits within the screen; the academic depth follows below.
    _render_embed()

    st.divider()
    _render_academic_framing()
    _render_quickstart()
    st.divider()
    _render_related_links()


if __name__ == "__main__":
    main()
