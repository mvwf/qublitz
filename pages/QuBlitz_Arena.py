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


@st.cache_data(show_spinner=False)
def _load_game_html() -> str:
    return _GAME_HTML.read_text(encoding="utf-8")


def _sage_proxy_url() -> str:
    """Server-side Sage proxy URL, or '' for the offline heuristic Sage.

    Read from the QB_SAGE_PROXY_URL env var, falling back to st.secrets. The
    Anthropic key is never read here — only the proxy URL — so no key can leak
    into the client HTML.
    """
    url = os.environ.get("QB_SAGE_PROXY_URL", "")
    if not url:
        try:
            url = st.secrets.get("QB_SAGE_PROXY_URL", "")  # type: ignore[attr-defined]
        except Exception:
            url = ""
    return str(url or "")


def _render_academic_framing():
    st.markdown(
        "**QuBlitz** is a quantum tactics game where every unit on the board is a **qubit**, "
        "every action is a **quantum gate**, and combat resolves via the **Born rule**. It is "
        "the interactive companion to this simulator: the same T₁/T₂ physics you explore on the "
        "other pages drive the game's decoherence — so playing builds intuition for the math."
    )

    with st.expander("🎯 Learning objectives — what you'll reason about", expanded=True):
        st.markdown(
            "- **Superposition & state preparation** — drive a qubit between |0⟩, |1⟩, and |+⟩.\n"
            "- **The Born rule** — measurement outcomes are probabilistic in the qubit's amplitudes.\n"
            "- **Bloch-sphere state** — read a live qubit as a vector (the panel shows it each turn).\n"
            "- **T₁/T₂ decoherence** — watch charge relax exponentially when a unit sits idle.\n"
            "- **Relative phase** — discover why Z/S/T gates matter only through interference.\n"
            "- **Entanglement & the no-communication theorem** — Bell pairs that collapse together."
        )

    with st.expander("🧭 Concept map — every action maps to real quantum mechanics"):
        st.markdown(
            "| In-game action | Quantum concept | In-engine rule |\n"
            "|---|---|---|\n"
            "| Charge with **X** / **H** | State preparation, superposition | X → |1⟩ (100%); H → |+⟩ (P(1)=50%) |\n"
            "| **Attack** | Born rule | P(hit) = charge = P(|1⟩); a target in |1⟩ takes a CRITICAL |\n"
            "| **Idle** a turn | T₁/T₂ relaxation (Lindblad) | charge ∝ e^(−t/T₁), coherence ∝ e^(−t/T₂) |\n"
            "| **GUARD** | X-basis measurement, relative phase | crit risk = (1−x)/2; |+⟩ safe, |−⟩ exposed |\n"
            "| **Z / S / T** | relative-phase rotation | invisible to charge — matters only via the GUARD's X-basis |\n"
            "| **CNOT** | entanglement (Bell pair) | |Φ⁺⟩; measuring one collapses both; no faster-than-light signalling |\n"
            "| **MEASURE** | projective collapse | |ψ⟩ → |0⟩ or |1⟩, then stabilizes the unit |\n"
        )

    with st.expander("🔬 The physics is real — and unit-tested"):
        st.markdown(
            "QuBlitz is not hand-waving flavor. Each unit evolves under the **exact Lindblad "
            "master-equation solution** for amplitude damping (T₁) plus dephasing (T₂), the same "
            "open-quantum-system model behind the simulator. Open the in-game **Physics Lab** to "
            "inspect a live density matrix ρ. The engine ships with a regression suite that asserts:\n\n"
            "- decoherence is **exponential** (R² > 0.99) and the fit recovers the input T₁ to within 1%,\n"
            "- a CNOT makes a true Bell pair with **perfectly correlated** collapse, and\n"
            "- a local gate on one half **never** changes the partner's marginal (no-communication)."
        )

    with st.expander("🎓 Research context"):
        st.markdown(
            "Built by **David Mukuruva** within the **Fitzpatrick Lab, Dartmouth College** "
            "(PI: Prof. Mattias Fitzpatrick), as the game companion to the lab's qubit simulator. "
            "The roadmap calibrates the game's decoherence to **real measured T₁/T₂**, connecting "
            "the proposal's research questions — mapping BCTDS parameters to T₂ degradation, and "
            "characterizing non-Markovian (HEOM) memory effects — to an interactive learning tool."
        )


def main():
    st.set_page_config(page_title="QuBlitz Arena", layout="wide")

    st.sidebar.image(load_logo("images/qublitz.png"))
    st.sidebar.image(load_logo("images/logo.png"))
    st.sidebar.markdown(
        '<div style="text-align:center;"><a href="https://sites.google.com/view/fitzlab/home" '
        'target="_blank" style="font-size:1.2rem; font-weight:bold;">FitzLab Website</a></div>',
        unsafe_allow_html=True,
    )

    st.title("⚔️ QuBlitz Arena")
    _render_academic_framing()

    st.markdown("### ▶ Play")
    st.caption("Click the board, then: A=attack · H/X/Y/Z=charge · M=measure · C=CNOT · Space=end turn. Press ? for the Sage.")

    proxy = _sage_proxy_url()
    inject = f"<script>window.QB_SAGE_PROXY={json.dumps(proxy)};</script>" if proxy else ""
    # Height covers the title bar + 800px board + side panels + event log.
    st.components.v1.html(inject + _load_game_html(), height=1100, scrolling=True)

    if proxy:
        st.caption("✦ Live Claude Sage connected via a server-side proxy (the API key stays on the proxy).")
    else:
        st.caption(
            "The offline heuristic Sage is active — it gives concept-linked, per-unit advice with no "
            "API key. To enable the live Claude Sage, set `QB_SAGE_PROXY_URL` in the app secrets."
        )


if __name__ == "__main__":
    main()
