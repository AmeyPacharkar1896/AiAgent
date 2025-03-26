import os
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import types

def load_training_data(csv_path):
    """
    Reads the CSV file from csv_path (expected to have 'input' and 'output' columns),
    converts each row into a TuningExample, and returns a TuningDataset.
    """
    try:
        df = pd.read_csv(csv_path, engine='python', encoding='utf-8')
        print(f"Loaded {len(df)} examples from {csv_path}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        raise

    # Create a list of TuningExample objects from each row.
    examples = [
        types.TuningExample(
            text_input=row["input"],
            output=row["output"],
        )
        for row in df.to_dict(orient='records')
    ]
    return types.TuningDataset(examples=examples)

def main():
    # Load environment variables from the .env file.
    load_dotenv()
    API_KEY = os.getenv("API_KEY")
    if not API_KEY:
        print("API_KEY not found in .env file. Please set your Gemini API key there.")
        return

    # Configure the Gemini API client.
    genai.configure(api_key=API_KEY)
    client = genai.Client()

    # Define the path to your final_data.csv inside the data folder.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "data", "final_data.csv")
    
    # Load training examples from the CSV file.
    training_dataset = load_training_data(csv_path)

    # Create the tuning configuration.
    tuning_config = types.CreateTuningJobConfig(
        epoch_count=5,
        batch_size=4,
        learning_rate=0.001,
        tuned_model_display_name="test tuned model"
    )

    # Submit the tuning job.
    try:
        tuning_job = client.tunings.tune(
            base_model='models/gemini-1.5-flash-001-tuning',
            training_dataset=training_dataset,
            config=tuning_config
        )
        print("Tuning job created successfully:")
        print(tuning_job)
    except Exception as e:
        print("Failed to create tuning job:", e)
        return

    # Generate content with the tuned model.
    try:
        response = client.models.generate_content(
            model=tuning_job.tuned_model.model,
            contents='III'
        )
        print("Response from tuned model:")
        print(response.text)
    except Exception as e:
        print("Failed to generate content:", e)

if __name__ == "__main__":
    main()
