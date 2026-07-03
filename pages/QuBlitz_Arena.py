"""QuBlitz Arena — David Mukuruva's contribution to the Fitzpatrick Lab platform.

Embeds the QuBlitz battle game (a self-contained quantum_chess.html) and frames
it as an academic learning tool: every game action maps to a real quantum
concept, and the in-engine physics is the same unit-tested open-quantum-system
model the simulator pages use.

Sage AI: the embedded game's Sage runs on a smart offline heuristic out of the
box (no key, nothing to configure). If a server-side proxy URL is provided via
``st.secrets["QB_SAGE_PROXY_URL"]`` (or the env var of the same name), the live
Claude Sage is enabled by injecting it as ``window.QB_SAGE_PROXY`` — the API key
itself lives only on that proxy, never in this page or the client HTML.
"""
import json
import os
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


def _has_secrets_file() -> bool:
    """True only if a Streamlit secrets.toml actually exists.

    Accessing ``st.secrets`` with no secrets file makes Streamlit render a red
    'No secrets found' error in the app, so we check first and skip it when none
    is configured (the heuristic Sage is the no-secret default).
    """
    candidates = [
        Path.cwd() / ".streamlit" / "secrets.toml",
        Path.home() / ".streamlit" / "secrets.toml",
    ]
    return any(c.exists() for c in candidates)


def _sage_proxy_url() -> str:
    """Server-side Sage proxy URL, or '' for the offline heuristic Sage.

    Read from the QB_SAGE_PROXY_URL env var, falling back to st.secrets only when
    a secrets file exists. The Anthropic key is never read here — only the proxy
    URL — so no key can leak into the client HTML.
    """
    url = os.environ.get("QB_SAGE_PROXY_URL", "")
    if not url and _has_secrets_file():
        try:
            url = st.secrets.get("QB_SAGE_PROXY_URL", "")  # type: ignore[attr-defined]
        except Exception:
            url = ""
    return str(url or "")


def _render_sidebar():
    st.sidebar.image(load_logo("images/qublitz.png"))
    st.sidebar.image(load_logo("images/logo.png"))
    st.sidebar.markdown(
        '<div style="text-align:center;"><a href="https://sites.google.com/view/fitzlab/home" '
        'target="_blank" style="font-size:1.2rem; font-weight:bold;">FitzLab Website</a></div>',
        unsafe_allow_html=True,
    )


def _render_embed():
    proxy = _sage_proxy_url()
    inject = f"<script>window.QB_SAGE_PROXY={json.dumps(proxy)};</script>" if proxy else ""
    st.components.v1.html(inject + _load_game_html(), height=_EMBED_HEIGHT, scrolling=True)
    if proxy:
        st.success(
            "Live Claude Sage connected via a server-side proxy — the API key stays on the "
            "proxy, never in this page.",
            icon=":material/verified:",
        )
    else:
        st.info(
            "The offline heuristic Sage is active — concept-linked, per-unit advice with no API "
            "key. To enable the live Claude Sage, set `QB_SAGE_PROXY_URL` in the app secrets.",
            icon=":material/auto_awesome:",
        )


def _render_academic_framing():
    st.markdown(
        "**QuBlitz** is a quantum tactics game where every unit on the board is a **qubit**, "
        "every action is a **quantum gate**, and combat resolves via the **Born rule**. It is "
        "the interactive companion to this simulator: the same T₁/T₂ physics you explore on the "
        "other pages drive the game's decoherence — so playing builds intuition for the math."
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
    st.markdown("**Keep exploring the platform**")
    col1, col2 = st.columns(2)
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
    except Exception:
        with col1:
            st.markdown("**New to qubits?** Open the *Quantum Measurement Tutorial* page.")
        with col2:
            st.markdown("**Want the real physics?** Open the *Qubit Simulator* page.")


def main():
    st.set_page_config(page_title="QuBlitz Arena", layout="wide")
    _render_sidebar()

    st.title("QuBlitz Arena")
    st.caption(
        "Learn quantum mechanics by playing it — every unit is a qubit, every move is a gate.  "
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
