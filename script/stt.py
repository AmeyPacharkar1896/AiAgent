import os
import whisper
import sounddevice as sd
import numpy as np

# Define the folder to store the model files
MODEL_DIR = "./models"

# Create the folder if it doesn't exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Load the Whisper model using the custom download root
# This will save the model to the 'models' folder on first run,
# and use the cached model on subsequent runs.
model = whisper.load_model("small", download_root=MODEL_DIR)

def record_audio(duration=5, fs=16000):
    """Record audio for a given duration (in seconds) at the specified sample rate."""
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Wait until the recording is finished
    return np.squeeze(audio)

def transcribe_audio(audio_array):
    """Transcribe recorded audio using Whisper."""
    result = model.transcribe(audio_array, fp16=False)
    return result["text"]
