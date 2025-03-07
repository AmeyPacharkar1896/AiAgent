import os
from TTS.api import TTS
import simpleaudio as sa

# Load a pre-trained TTS model (ensuring CPU mode)
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True, gpu=False)

def synthesize_speech(text, output_path="./output/response.wav"):
    """Convert text to speech and save the output as a WAV file."""
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Synthesizing speech...")
    tts.tts_to_file(text=text, file_path=output_path)
    
    # Check if the file was saved
    if os.path.exists(output_path):
        print(f"Audio file successfully saved: {output_path}")
    else:
        print("Error: Audio file was not saved.")

def play_audio(file_path="./output/response.wav"):
    """Play the audio from a given file path."""
    if not os.path.exists(file_path):
        print("Audio file does not exist:", file_path)
        return
    print("Playing audio file:", file_path)
    try:
        wave_obj = sa.WaveObject.from_wave_file(file_path)
        play_obj = wave_obj.play()
        play_obj.wait_done()
        print("Playback finished.")
    except Exception as e:
        print("Error playing audio:", e)
