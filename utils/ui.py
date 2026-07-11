"""utils/ui.py — UP-1: the page contract every page starts with.

Before this, 4 of 9 pages under pages/ called st.set_page_config ad hoc
(inconsistent titles, some with no icon) and the other 5 didn't call it at
all — no page had a header a student could recognize as "the same site."
page_header() is now the one shared call every page makes first, in place
of its own st.set_page_config + st.title. See docs/PAGE_CONTRACT.md.
"""
import streamlit as st

from utils.branding import load_logo


def page_header(title: str, icon: str, blurb: str, group: str) -> None:
    """Set the page config and render a consistent header row.

    Must be the FIRST Streamlit call on the page — page_header() itself
    calls st.set_page_config, which Streamlit requires to run before any
    other st.* call, same rule as calling st.set_page_config directly.

    title: the page's own name, shown in the header and the browser tab.
    icon: a single emoji, shown in the header and as the browser-tab icon.
    blurb: one sentence naming the actual physics/action on this page —
        specificity over polish (a generic blurb is what makes a page read
        as AI-generated filler, not a chatbot's absence).
    group: which of the landing page's three sections this page belongs to
        (Learn / Play / Research tools) — not rendered as a nav element
        yet (UP-2, the landing rewrite, is still pending maintainer sign-off
        per this file's own history), just shown as a small caption prefix
        so the eventual grouping is already decided and visible per-page.
    """
    st.set_page_config(page_title=f"{title} — QuBlitz", page_icon=icon, layout="wide")
    col_logo, col_text = st.columns([1, 14])
    with col_logo:
        try:
            st.image(load_logo("images/logo.png"), width=44)
        except Exception:
            pass
    with col_text:
        # st.title (not st.markdown("##...")) — several pages already called
        # st.title directly, and at least one existing test
        # (test_qublitz_arena.py) specifically asserts against Streamlit's
        # st.title widget type, not markdown headers. Matching that instead
        # of quietly breaking it is also just the more literal "contract."
        st.title(f"{icon} {title}")
        st.caption(f"{group} · {blurb}")
    st.divider()
