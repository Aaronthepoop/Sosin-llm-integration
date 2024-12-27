
import torch
from openai import OpenAI, OpenAIError, RateLimitError
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import os
import time
from transformers import BertForQuestionAnswering, BertTokenizer, T5ForConditionalGeneration, T5Tokenizer

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # Load the API key from environment variable
client = OpenAI(api_key='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')

# Constants
OUTPUT_DIR = "static"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def fine_tune_bert():
    """Initialize BERT for question answering."""
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    return model, tokenizer

def fine_tune_t5():
    """Initialize T5 for explanation generation."""
    model = T5ForConditionalGeneration.from_pretrained('t5-large')
    tokenizer = T5Tokenizer.from_pretrained('t5-large')
    return model, tokenizer

def generate_mcq_with_openai(prompt, retries=3):
    """Generate MCQ using OpenAI API."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4",  # Updated to GPT-4
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except RateLimitError as e:
            print(f"Rate limit exceeded: {e}. Retrying {retries - attempt - 1} more times.")
            time.sleep(60)
        except OpenAIError as e:
            print(f"OpenAI error: {e}")
            break
    return None

def generate_map():
    """Create a map using Matplotlib and Cartopy."""
    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([68, 97, 8, 37], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.RIVERS, edgecolor='blue')

    locations = {
        'Pataliputra': (85.144, 25.611),
        'Taxila': (72.836, 33.737),
        'Ujjain': (75.784, 23.179),
        'Kalinga': (85.833, 20.936)
    }
    for location, (lon, lat) in locations.items():
        ax.plot(lon, lat, 'ro', transform=ccrs.PlateCarree())
        ax.text(lon + 0.5, lat, location, transform=ccrs.PlateCarree())

    plt.title('Major Locations of the Maurya Kingdom')
    map_path = os.path.join(OUTPUT_DIR, "generated_map.png")
    plt.savefig(map_path)
    plt.close()
    return map_path

def generate_table():
    """Create a table and save as Excel file."""
    data = {
        'Location': ['Pataliputra', 'Taxila', 'Ujjain', 'Kalinga'],
        'Longitude': [85.144, 72.836, 75.784, 85.833],
        'Latitude': [25.611, 33.737, 23.179, 20.936]
    }
    df = pd.DataFrame(data)
    table_path = os.path.join(OUTPUT_DIR, "generated_table.xlsx")
    df.to_excel(table_path, index=False)
    return table_path

def classify_content_type(question, retries=3):
    """Classify the question into map, table, or image."""
    prompt = f"Classify the following question into one of the categories: map, table, image.\n\nQuestion: {question}\n\nContent type:"
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4",  # Updated to GPT-4
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            content_type = response.choices[0].message.content.strip().lower()
            print(f"Debug: classify_content_type output: {content_type}")  # Debug Statement
            return content_type
            print(f"Rate limit exceeded: {e}. Retrying {retries - attempt - 1} more times.")
            time.sleep(60)
        except OpenAIError as e:
            print(f"OpenAI error: {e}")
            break
    return "unknown"

def create_mcq_with_content(question, options, explanation, content_type):
    """Generate content based on classified type and save the MCQ."""
    if content_type == 'map':
        content_path = generate_map()
    elif content_type == 'table':
        content_path = generate_table()
    else:
        print(f"Error: Unsupported content type '{content_type}'")  # Debug Statement
        raise ValueError("Unsupported content type")

    mcq_path = os.path.join(OUTPUT_DIR, "mcq_question.txt")
    with open(mcq_path, 'w') as f:
        f.write(f"Question: {question}\n")
        for option in options:
            f.write(f"{option}\n")
        f.write(f"Explanation: {explanation}\n")
        f.write(f"Content Path: {content_path}\n")

    return mcq_path, content_path

def main():
    # Initialize models
    model_bert, tokenizer_bert = fine_tune_bert()
    model_t5, tokenizer_t5 = fine_tune_t5()

    # Generate MCQ using OpenAI
    prompt = "Generate a multiple-choice question about climate change."
    mcq = generate_mcq_with_openai(prompt)
    print(f"Generated MCQ: {mcq}")

    # Example MCQ question and options
    question = "Which of the following locations was the capital of the Maurya Kingdom?"
    options = ["A) Pataliputra", "B) Taxila", "C) Ujjain", "D) Kalinga"]
    explanation = "Pataliputra was the capital of the Maurya Kingdom, as shown on the map."

    # Classify content type
    content_type = classify_content_type(question)
    print(f"Classified content type: {content_type}")

    # Create MCQ with associated content
    print(f"Debug: Classified content type is '{content_type}'")  # Debug Statement
    mcq_path, content_path = create_mcq_with_content(question, options, explanation, content_type)
    print(f"MCQ saved to {mcq_path}")
    print(f"Content saved to {content_path}")

if __name__ == "__main__":
    main()
