import re
import csv
import os
from datetime import datetime

# Regular expression pattern to match WhatsApp chat messages.
# Expected format: "31/05/2023, 9:07 pm - Amey Pacharkar: Or only other branches?"
pattern = re.compile(r'^(\d{2}\/\d{2}\/\d{4},\s*\d{1,2}:\d{2}\s*[ap]m)\s*-\s*(.*?):\s(.*)$')

def parse_chat(file_path):
    """
    Parses the WhatsApp chat log file and returns a list of structured messages.
    Each message is a dict with 'timestamp', 'sender', and 'message' keys.
    """
    messages = []
    current_message = None
    print(f"Processing file: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            match = pattern.match(line)
            if match:
                # If there is an existing message, finish it up
                if current_message:
                    # Trim whitespace for sender and message
                    current_message["sender"] = current_message["sender"].strip()
                    current_message["message"] = current_message["message"].strip()
                    # Only add the message if it doesn't equal "<Media omitted>"
                    if current_message["message"] != "<Media omitted>":
                        messages.append(current_message)
                timestamp_str, sender, message = match.groups()
                # Convert timestamp string to ISO format
                try:
                    timestamp = datetime.strptime(timestamp_str, "%d/%m/%Y, %I:%M %p")
                    timestamp_iso = timestamp.isoformat()
                except ValueError:
                    timestamp_iso = timestamp_str  # Retain original if conversion fails
                current_message = {
                    "timestamp": timestamp_iso,
                    "sender": sender,
                    "message": message
                }
            else:
                # Handle multi-line messages by appending the line
                if current_message:
                    current_message["message"] += " " + line

        # Add the final message if available
        if current_message:
            current_message["sender"] = current_message["sender"].strip()
            current_message["message"] = current_message["message"].strip()
            if current_message["message"] != "<Media omitted>":
                messages.append(current_message)
    
    return messages

def write_chunks_to_csv(messages, input_file, chunk_size=1000, output_prefix='processed_chat_chunk'):
    """
    Divides the messages into chunks and saves each chunk to a separate CSV file
    in the same folder as the input text file.
    """
    total_messages = len(messages)
    input_dir = os.path.dirname(input_file)
    
    for i in range(0, total_messages, chunk_size):
        chunk = messages[i:i+chunk_size]
        chunk_index = i // chunk_size + 1
        output_file = os.path.join(input_dir, f"{output_prefix}_{chunk_index}.csv")
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["timestamp", "sender", "message"])
            writer.writeheader()
            for msg in chunk:
                writer.writerow(msg)
        print(f"Saved chunk {chunk_index} with {len(chunk)} messages to {output_file}")

if __name__ == '__main__':
    # Use a raw string for Windows file paths to avoid escape issues.
    input_file = r'D:\ai_agent\data\chat_logs\swaraj\WhatsApp_Chat_with_Swaraj.txt'  
    processed_messages = parse_chat(input_file)
    print(f"Total processed messages: {len(processed_messages)}")
    
    # Define how many messages per CSV chunk (adjust as needed)
    chunk_size = 1000
    write_chunks_to_csv(processed_messages, input_file, chunk_size=chunk_size)
