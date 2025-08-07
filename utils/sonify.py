import numpy as np
from PIL import Image
import soundfile as sf
import tempfile

def generate_audio(image, min_x, max_x, min_y, max_y):
    # Accepts a PIL Image object
    data = np.array(image.convert('L'), dtype=np.float32) / 255.0

    # Generate a simple sine wave based on image mean (placeholder logic)
    duration = max_x-min_x  # seconds
    sr = 22050
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freq = min_y + (max_y - min_y) * np.mean(data)
    audio = np.sin(2*np.pi*freq * t)

    #modulate 

    # Save to temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        sf.write(tmp_wav.name, audio, sr)
        return tmp_wav.name