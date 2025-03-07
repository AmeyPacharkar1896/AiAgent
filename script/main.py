import asyncio
from concurrent.futures import ThreadPoolExecutor
from stt import record_audio, transcribe_audio
from nlp import generate_response
from tts import synthesize_speech, play_audio

# Create a ThreadPoolExecutor to run blocking functions asynchronously
executor = ThreadPoolExecutor(max_workers=4)

async def run_in_executor(func, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, func, *args)

async def conversation_loop_async():
    print("Starting async conversation loop with context buffering. Say 'exit' to quit.")
    
    # System message to guide the agent's responses
    system_message = "You are an AI assistant that provides helpful, friendly, and creative responses. Do not simply repeat what the user says."

    conversation_history = []
    max_history_turns = 6

    while True:
        # Record audio asynchronously
        print("\nPlease speak something (5-second recording):")
        audio_clip = await run_in_executor(record_audio, 5)
        
        # Transcribe asynchronously
        transcript = await run_in_executor(transcribe_audio, audio_clip)
        print("You said:", transcript)
        
        # Exit condition
        if transcript.strip().lower() == "exit":
            print("Exiting conversation.")
            break

        # Append user's input to the conversation history
        conversation_history.append("User: " + transcript)
        
        # Prepare context buffer from recent turns
        context_buffer = "\n".join(conversation_history[-max_history_turns:])
        
        # Construct prompt with a system message for guidance
        prompt = system_message + "\n" + context_buffer + "\nAgent:"
        
        # Generate a response asynchronously
        response_text = await run_in_executor(generate_response, prompt)
        print("Agent response:", response_text)
        
        # Append agent's response to conversation history for context
        conversation_history.append("Agent: " + response_text)
        
        # Synthesize speech asynchronously
        await run_in_executor(synthesize_speech, response_text, "./output/response.wav")
        
        # Play the generated audio asynchronously (this is sequential, so agent listens only after speaking)
        await run_in_executor(play_audio, "./output/response.wav")

if __name__ == "__main__":
    asyncio.run(conversation_loop_async())
