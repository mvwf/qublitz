import streamlit as st


def main():
    st.set_page_config(page_title="QuBlitz Arena", layout="wide")

    st.title("⚔️ QuBlitz Arena")

    st.markdown(
        "**QuBlitz** is a quantum tactics game where every unit on the board is a qubit, "
        "every action is a quantum gate, and combat resolves via the Born rule. Charge a "
        "qubit (H/X), close in, and strike — a target caught in |1⟩ takes a critical hit, "
        "while decoherence bleeds your charge away each turn. It is the game companion to "
        "this simulator: the same T₁/T₂ physics you explore here drive the game's "
        "decoherence model."
    )

    st.info("🎮 Game loading coming in C02 — the playable board will embed here next cycle.")


if __name__ == "__main__":
    main()
