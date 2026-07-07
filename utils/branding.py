"""Shared, cached loaders for the static sidebar branding images.

Every page used to call ``Image.open("images/qublitz.png")`` and
``Image.open("images/logo.png")`` in its sidebar on *every* Streamlit rerun,
re-decoding the PNGs from disk each time. ``@st.cache_resource`` decodes them
once per process and hands back the same object thereafter.
"""
import streamlit as st
from PIL import Image


@st.cache_resource(show_spinner=False)
def load_logo(path: str) -> Image.Image:
    return Image.open(path)
