
import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import librosa

sound_folder = 'sound_files'
output_folder = '../images'
os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(sound_folder):
    if file.endswith('.wav') or file.endswith('.mp3'):
        file_path = os.path.join(sound_folder, file)
        if file.endswith('.wav'):
            audio, sr = sf.read(file_path)
        else:
            audio, sr = librosa.load(file_path, sr=None, mono=False)
            # librosa returns (n,) or (2, n) for mono/stereo, so transpose if stereo
            if audio.ndim > 1:
                audio = audio[0]  # Use first channel

        # Use only one channel if stereo
        if audio.ndim > 1:
            audio = audio[:, 0]

        # Normalize amplitude to [0,1]
        audio_norm = (audio - audio.min()) / (audio.max() - audio.min() + 1e-12)

        # Duplicate single amplitude row vertically to create 2D image
        n_rows = 64
        img_array = np.tile(audio_norm, (n_rows, 1))

        # Save as PNG (lossless)
        plt.imsave(os.path.join(output_folder, f'{os.path.splitext(file)[0]}.png'),
                   img_array, cmap='gray', vmin=0, vmax=1)

        print(f"Converted {file} to image with shape {img_array.shape}")