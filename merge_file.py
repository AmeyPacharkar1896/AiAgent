import csv
import os

# Define maximum allowed characters per example
MAX_INPUT_CHARS = 40000
MAX_OUTPUT_CHARS = 5000

def merge_csv_files(chat_logs_dir):
    """
    Recursively walks through chat_logs_dir and its subfolders to merge all CSV chunk files.
    Returns a list of dictionaries with keys "input" and "output".
    """
    merged_rows = []
    for subdir, dirs, files in os.walk(chat_logs_dir):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(subdir, file)
                print(f"Processing CSV: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        merged_rows.append(row)
    return merged_rows

def parse_qna_file(qna_file_path):
    """
    Parses the QnA.txt file where the format is:
    
    Q: Can you tell me a little about yourself?
    A: Hello, I have a background of computer engineering...
    
    Returns a list of dictionaries with keys "input" and "output".
    """
    qna_pairs = []
    with open(qna_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    question, answer = None, None
    for line in lines:
        line = line.strip()
        if line.startswith("Q:"):
            # Save previous pair if exists
            if question is not None and answer is not None:
                qna_pairs.append({"input": question, "output": answer})
                question, answer = None, None
            question = line[2:].strip()
        elif line.startswith("A:"):
            answer = line[2:].strip()
            if question is not None:
                qna_pairs.append({"input": question, "output": answer})
                question, answer = None, None
        else:
            # Append multi-line content if applicable
            if answer is not None:
                answer += " " + line
            elif question is not None:
                question += " " + line

    # Add any remaining pair
    if question is not None and answer is not None:
        qna_pairs.append({"input": question, "output": answer})
    
    print(f"Parsed {len(qna_pairs)} QnA pairs from {qna_file_path}")
    return qna_pairs

def merge_all_data(data_dir):
    """
    Merges data from CSV chunk files in data/chat_logs and the QnA text file,
    returning a single list of dialogue pairs.
    """
    chat_logs_dir = os.path.join(data_dir, "chat_logs")
    qna_file_path = os.path.join(data_dir, "QnA", "QnA.txt")
    
    merged_data = []
    
    # Merge CSV chat logs data.
    csv_data = merge_csv_files(chat_logs_dir)
    print(f"Total CSV rows: {len(csv_data)}")
    merged_data.extend(csv_data)
    
    # Parse QnA file.
    qna_data = parse_qna_file(qna_file_path)
    print(f"Total QnA pairs: {len(qna_data)}")
    merged_data.extend(qna_data)
    
    return merged_data

def enforce_character_limits(row):
    """
    Truncates the input and output fields of a row if they exceed the limits.
    Returns the modified row.
    """
    if len(row["input"]) >= MAX_INPUT_CHARS:
        row["input"] = row["input"][:MAX_INPUT_CHARS - 1]
    if len(row["output"]) >= MAX_OUTPUT_CHARS:
        row["output"] = row["output"][:MAX_OUTPUT_CHARS - 1]
    return row

def write_final_csv(data_rows, final_csv_path):
    """
    Writes all merged data rows into a final CSV file.
    Enforces that the input has fewer than 40,000 characters and the output fewer than 5,000 characters.
    """
    with open(final_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['input', 'output']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in data_rows:
            row = enforce_character_limits(row)
            writer.writerow(row)
    
    print(f"Merged a total of {len(data_rows)} rows into {final_csv_path}")

if __name__ == '__main__':
    # Define the base data directory.
    data_dir = r"data"
    final_csv_path = os.path.join(data_dir, "final_data.csv")
    
    # Merge data from chat logs and QnA.txt.
    all_data = merge_all_data(data_dir)
    
    # Write the final merged data to final_data.csv with enforced limits.
    write_final_csv(all_data, final_csv_path)
