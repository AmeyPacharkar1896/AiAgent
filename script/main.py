from stt import record_audio, transcribe_audio
from nlp import generate_response
from tts import synthesize_speech, play_audio

def conversation_loop():
    print("Starting conversation loop. Say 'exit' to quit.")
    while True:
        print("\nPlease speak something (5-second recording):")
        audio_clip = record_audio(duration=5)
        transcript = transcribe_audio(audio_clip)
        print("You said:", transcript)
        
        # Check if the user said "exit" to quit the conversation
        if transcript.strip().lower() == "exit":
            print("Exiting conversation.")
            break
        
        # Generate a response using the language model
        response_text = generate_response(transcript)
        print("Agent response:", response_text)
        
        # Convert the response text to speech and play it
        synthesize_speech(response_text, output_path="./output/response.wav")
        play_audio("output/response.wav")

if __name__ == "__main__":
    conversation_loop()
