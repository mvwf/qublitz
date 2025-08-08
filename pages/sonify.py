import streamlit as st
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import soundfile as sf
import tempfile
import json
import os

DEFAULT_SR = 22050

# ——— Helper functions ——————————————————————————————————

def get_time_axis(cols, min_x, max_x):
    return np.linspace(min_x, max_x, cols, endpoint=False)

def generate_modulated_matrix(env, t, y_freqs):
    rows, cols = env.shape
    mod = np.zeros_like(env, dtype=float)
    for i in range(rows):
        mod[i, :] = env[i, :] * np.cos(2 * np.pi * y_freqs[i] * t)
    return mod

def compute_fft_map(mat, sample_spacing):
    fft_r = np.fft.rfft(mat, axis=1)
    mag = np.abs(fft_r)
    freqs = np.fft.rfftfreq(mat.shape[1], d=sample_spacing)
    return mag, freqs

def generate_audio(env, t_env, row_idx, carrier_freq, duration, sr=DEFAULT_SR):
    cols = env.shape[1]
    # audio time axis
    N = int(np.ceil(duration * sr))
    t_audio = np.linspace(0, duration, N, endpoint=False)
    # interpolate envelope
    low_env = env[row_idx, :]
    high_env = np.interp(t_audio, t_env - t_env[0], low_env)
    # carrier
    carrier = np.cos(2 * np.pi * carrier_freq * t_audio)
    audio = high_env * carrier
    audio /= (np.max(np.abs(audio)) + 1e-12)
    # write WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio.astype(np.float32), sr)
        return tmp.name, audio

# ——— Streamlit app ——————————————————————————————————

st.set_page_config(layout="wide")

def main():
    st.title("Image Sonification, Turn Images into Sound!")
    qublitz_logo = Image.open("images/qublitz.png")
    st.sidebar.image(qublitz_logo)
    logo = Image.open("images/logo.png") 
    st.sidebar.image(logo) # display logo on the side 
    st.sidebar.markdown('<div style="text-align:center;"><a href="https://sites.google.com/view/fitzlab/home" target="_blank" style="font-size:1.2rem; font-weight:bold;">FitzLab Website</a></div>', unsafe_allow_html=True)
    # --- Load image list for premade option (not shown in sidebar) ---
    helper_path = os.path.join("sonify_images", "image_helper.json")
    image_list = []
    try:
        with open(helper_path, "r") as f:
            image_list = json.load(f)
    except Exception:
        pass


    # --- 1) Image selection: premade or upload ---

    st.markdown("### Choose an Image")
    image_mode = st.radio("Select image source:", ["Upload your own", "Preloaded"], horizontal=True, index=0)


    img_original = None
    img_info = None
    reset_image = False
    if image_mode == "Premade":
        premade_names = [entry['filename'] for entry in image_list]
        selected_name = st.selectbox("Select a premade image:", premade_names, key="premade_select")
        selected_entry = next((e for e in image_list if e['filename'] == selected_name), None)
        if selected_entry:
            img_path = os.path.join("sonify_images", selected_entry['filename'])
            img_original = Image.open(img_path)
            img_info = selected_entry
            reset_image = True
    else:
        uploaded = st.file_uploader("Upload PNG or JPG", type=["png", "jpg"], key="uploader")
        if uploaded:
            img_original = Image.open(uploaded)
            reset_image = True

    if img_original is None:
        st.info("Please select or upload an image to begin.")
        return

    # Convert to grayscale for processing
    img_gray = img_original.convert("L")
    gray0 = np.array(img_gray, dtype=float) / 255.0

    # Always reset the image matrix when a new image is selected/uploaded
    if reset_image or 'img_mat' not in st.session_state:
        st.session_state['img_mat'] = gray0.copy()

    # Display actual original image (color or grayscale)
    st.markdown("### Original Image")
    st.image(img_original)
    if img_info:
        st.caption(img_info.get('text', ''))
        if img_info.get('link'):
            st.markdown(f"[Source]({img_info['link']})")

    # Convert to grayscale for processing
    img_gray = img_original.convert("L")
    gray0 = np.array(img_gray, dtype=float) / 255.0
    # gray0 = np.flipud(gray0)    # make row=0 the bottom

    if 'img_mat' not in st.session_state:
        st.session_state['img_mat'] = gray0.copy()

    # --- 2) Sidebar: axis controls ---
    st.sidebar.markdown("## Axis Controls")
    min_x = 0.0
    max_x = st.sidebar.number_input("Time Duration [s]", value=4.0, step=0.1, min_value=0.1)

    # Single modulation frequency for all rows
    mod_freq = st.sidebar.number_input("Modulation Frequency [Hz]", value=50.0, step=1.0, min_value=0.0)

    sel_row = st.sidebar.number_input("Select Row", min_value=0,
                                      max_value=gray0.shape[0] - 1,
                                      value=gray0.shape[0] // 2, step=1)

    # --- 3) Sidebar: image permutations ---
    st.sidebar.markdown("## Image Permutations")
    if st.sidebar.button("Flip Horizontally"):
        st.session_state['img_mat'] = np.fliplr(st.session_state['img_mat'])
    if st.sidebar.button("Reset Image"):
        st.session_state['img_mat'] = gray0.copy()

    # --- 4) Derive everything from the current image matrix ---
    env = st.session_state['img_mat']
    rows, cols = env.shape
    t_lowres = get_time_axis(cols, min_x, max_x)

    # Set same modulation frequency for all rows
    y_freqs = np.full(rows, mod_freq)
    y_axis = np.arange(rows)  # For plotting

    sample_spacing = (max_x - min_x) / cols

    # --- 5) Display Grayscale Heatmap of the current image ---
    st.markdown("### Grayscale Heatmap")
    fig_gray = go.Figure(go.Heatmap(
        z=env,
        x=np.arange(cols), y=np.arange(rows),
        colorscale="gray", colorbar=dict(title="amp"),
        zmin=0, zmax=1
    ))
    fig_gray.update_xaxes(title="Column")
    fig_gray.update_yaxes(title="Row", autorange='reversed')
    fig_gray.add_hline(y=sel_row, line=dict(color="black", dash="dash"))
    st.plotly_chart(fig_gray, key="gray_heatmap")

    mod_mat = generate_modulated_matrix(env, t_lowres, y_freqs)

   
    # 5) 1-D signal and the modulated signal 
    st.markdown(f"### 1-D Signal for Row {sel_row}")
    sel_sig = env[sel_row, :]
    fig6_1d_signal = go.Figure(go.Scatter(x=t_lowres, y=sel_sig, mode='lines'))
    fig6_1d_signal.update_layout(
        title=f"Row {sel_row} Signal",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude"
    )
    st.plotly_chart(fig6_1d_signal, key="oned_signal")

    st.markdown(f"### Modulated Signal for Row {sel_row}")
    sel_mod_sig = mod_mat[sel_row, :]
    fig6_mod_signal = go.Figure(go.Scatter(x=t_lowres, y=sel_mod_sig, mode='lines'))
    fig6_mod_signal.update_layout(
        title=f"Row {sel_row} Modulated Signal (Modulation Frequency = {mod_freq} Hz)",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude"
    )
    st.plotly_chart(fig6_mod_signal, key="modulated_signal")

    st.markdown(f"### 1D FFT of Row {sel_row}")
    fft_s = np.abs(np.fft.rfft(sel_mod_sig))
    freq_s = np.fft.rfftfreq(cols, d=sample_spacing)
    fig6_fft_1d = go.Figure(go.Scatter(x=freq_s, y=fft_s, mode='lines'))
    fig6_fft_1d.update_layout(
        title=f"Peak @ ~{mod_freq:.1f} Hz (Modulation Frequency)",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude"
    )
    st.plotly_chart(fig6_fft_1d, key="fft_1d_signal")

    # 6) Audio Playback for that row
    st.markdown(f"### Audio for Row {sel_row}")
    wav_path, _ = generate_audio(env, t_lowres, sel_row,
                                 mod_freq,
                                 duration=max_x - min_x,
                                 sr=DEFAULT_SR)
    st.audio(wav_path, format="audio/wav")

if __name__ == "__main__":
    main()
