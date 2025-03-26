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
                # Save the previous message if it exists and isn't a media message
                if current_message:
                    current_message["sender"] = current_message["sender"].strip()
                    current_message["message"] = current_message["message"].strip()
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
                # Append to the previous message if it is a continuation of a multi-line message
                if current_message:
                    current_message["message"] += " " + line

        # Add the final message if available
        if current_message:
            current_message["sender"] = current_message["sender"].strip()
            current_message["message"] = current_message["message"].strip()
            if current_message["message"] != "<Media omitted>":
                messages.append(current_message)
    
    return messages

def group_messages_by_sender(messages):
    """
    Groups consecutive messages by the same sender.
    Returns a list of dicts with keys: 'sender' and 'text' (concatenated messages).
    """
    grouped = []
    if not messages:
        return grouped

    current_group = {"sender": messages[0]["sender"], "text": messages[0]["message"]}
    for msg in messages[1:]:
        if msg["sender"] == current_group["sender"]:
            current_group["text"] += " " + msg["message"]
        else:
            grouped.append(current_group)
            current_group = {"sender": msg["sender"], "text": msg["message"]}
    grouped.append(current_group)
    return grouped

def create_dialogue_pairs(grouped_messages, input_sender="Swaraj", output_sender="Amey"):
    """
    Creates dialogue pairs where a message group from input_sender is followed immediately by a group
    from output_sender.
    """
    pairs = []
    i = 0
    while i < len(grouped_messages) - 1:
        current = grouped_messages[i]
        next_group = grouped_messages[i+1]
        # Check if current group is from the input_sender and next is from the output_sender.
        if input_sender.lower() in current["sender"].lower() and output_sender.lower() in next_group["sender"].lower():
            pairs.append({
                "input": current["text"].strip(),
                "output": next_group["text"].strip()
            })
            i += 2  # Skip the next one as it has been paired
        else:
            i += 1
    return pairs

def write_chunks_to_csv(dialogue_pairs, input_file, chunk_size=1000, output_prefix='processed_chat_chunk'):
    """
    Divides the dialogue pairs into chunks and saves each chunk to a separate CSV file
    in the same folder as the input text file. The CSV will have columns 'input' and 'output'.
    """
    total_pairs = len(dialogue_pairs)
    input_dir = os.path.dirname(input_file)
    
    for i in range(0, total_pairs, chunk_size):
        chunk = dialogue_pairs[i:i+chunk_size]
        chunk_index = i // chunk_size + 1
        output_file = os.path.join(input_dir, f"{output_prefix}_{chunk_index}.csv")
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["input", "output"])
            writer.writeheader()
            for pair in chunk:
                writer.writerow(pair)
        print(f"Saved chunk {chunk_index} with {len(chunk)} dialogue pairs to {output_file}")

if __name__ == '__main__':
    # Use a raw string for Windows file paths to avoid escape issues.
    input_file = r'D:\ai_agent\data\chat_logs\swaraj\WhatsApp_Chat_with_Swaraj.txt'
    
    # Parse the chat messages.
    messages = parse_chat(input_file)
    print(f"Total processed messages: {len(messages)}")
    
    # Group consecutive messages by sender.
    grouped_messages = group_messages_by_sender(messages)
    print(f"Total grouped message segments: {len(grouped_messages)}")
    
    # Create dialogue pairs: Dovansh messages as input, followed by Amey messages as output.
    dialogue_pairs = create_dialogue_pairs(grouped_messages, input_sender="Swaraj", output_sender="Amey")
    print(f"Total dialogue pairs: {len(dialogue_pairs)}")
    
    # Define how many pairs per CSV chunk (adjust as needed)
    chunk_size = 1000
    write_chunks_to_csv(dialogue_pairs, input_file, chunk_size=chunk_size)
