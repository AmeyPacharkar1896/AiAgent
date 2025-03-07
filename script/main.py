from stt import record_audio, transcribe_audio
from nlp import generate_response
from tts import synthesize_speech, play_audio

def conversation_loop():
    print("Starting conversation loop with context buffering. Say 'exit' to quit.")
    # This list will hold the conversation history
    conversation_history = []
    max_history_turns = 6  # adjust this value as needed

    while True:
        print("\nPlease speak something (5-second recording):")
        audio_clip = record_audio(duration=5)
        transcript = transcribe_audio(audio_clip)
        print("You said:", transcript)

        # Exit condition
        if transcript.strip().lower() == "exit":
            print("Exiting conversation.")
            break

        # Append user input to the conversation history
        conversation_history.append("User: " + transcript)

        # Prepare the context buffer (limit to last max_history_turns exchanges)
        context_buffer = "\n".join(conversation_history[-max_history_turns:])

        # Construct a prompt that includes context and signals the agent to reply
        prompt = context_buffer + "\nAgent:"
        
        # Generate a response with context
        response_text = generate_response(prompt)
        print("Agent response:", response_text)

        # Append the agent's response to the history for context in future turns
        conversation_history.append("Agent: " + response_text)

        # Convert the agent's response to speech and play it
        synthesize_speech(response_text, output_path="output/response.wav")
        play_audio("output/response.wav")

if __name__ == "__main__":
    conversation_loop()
