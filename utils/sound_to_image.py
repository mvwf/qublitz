"""Convert a 1-D audio waveform into a 2-D grayscale image (row-tiled amplitude).

Was previously a one-off batch script only (`sound_files/*.wav|*.mp3` -> `../images/*.png`,
run manually to produce a few of the sonify_images/ gallery entries — e.g. the GW150914
waveform art). `audio_to_image_array` is the reusable half, now imported live by
`pages/Sonify.py`'s "Convert a sound file" mode so the same conversion runs in the app,
not just offline. The original batch entry point still works unchanged via `python -m
utils.sound_to_image` from the repo root, kept for regenerating the gallery assets.
"""

import numpy as np


def audio_to_image_array(audio, n_rows=64, max_cols=16384):
    """Normalize a 1-D (or stereo) waveform to [0,1] and tile it into an (n_rows, N) array.

    N is capped at max_cols via block-averaging, not the raw sample count: Streamlit's
    default JPEG encoder hard-fails above 65500px in either dimension (confirmed live —
    an 84672-sample clip crashed `st.image()` with "broken data stream when writing image
    file" before this cap existed), and a many-thousand-column Plotly figure is sluggish
    well before that limit anyway. 16384 matches the existing GW150914 gallery images'
    proven-safe width.
    """
    audio = np.asarray(audio, dtype=float)
    if audio.ndim > 1:
        # librosa.load(..., mono=False) returns (channels, N); soundfile returns (N, channels).
        audio = audio[0] if audio.shape[0] < audio.shape[1] else audio[:, 0]
    if len(audio) > max_cols:
        pad = (-len(audio)) % max_cols
        audio = np.pad(audio, (0, pad), mode="edge").reshape(max_cols, -1).mean(axis=1)
    span = audio.max() - audio.min()
    audio_norm = (audio - audio.min()) / (span + 1e-12) if span > 0 else np.full_like(audio, 0.5)
    return np.tile(audio_norm, (n_rows, 1))


if __name__ == "__main__":
    import os

    import librosa
    import matplotlib.pyplot as plt
    import soundfile as sf

    sound_folder = "sound_files"
    output_folder = "../images"
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(sound_folder):
        if file.endswith(".wav") or file.endswith(".mp3"):
            file_path = os.path.join(sound_folder, file)
            if file.endswith(".wav"):
                raw_audio, sr = sf.read(file_path)
            else:
                raw_audio, sr = librosa.load(file_path, sr=None, mono=False)

            img_array = audio_to_image_array(raw_audio)

            plt.imsave(
                os.path.join(output_folder, f"{os.path.splitext(file)[0]}.png"),
                img_array,
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            print(f"Converted {file} to image with shape {img_array.shape}")