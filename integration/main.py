import torch
from openai import OpenAI
import openai
from openai.error import OpenAIError
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import os
import time

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

# Test API connection
try:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Test API connection"}
        ]
    )
    print("API connection successful!")
    print("Response:", response.choices[0].message["content"])
except OpenAIError as e:
    print(f"OpenAI API error: {e}")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Constants
OUTPUT_DIR = "static"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def fine_tune_bert():
    """Initialize BERT for question answering."""
    from transformers import BertForQuestionAnswering, BertTokenizer
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    return model, tokenizer

def fine_tune_t5():
    """Initialize T5 for explanation generation."""
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    model = T5ForConditionalGeneration.from_pretrained('t5-large')
    tokenizer = T5Tokenizer.from_pretrained('t5-large')
    return model, tokenizer

def generate_mcq_with_openai(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except OpenAIError as e:
        print(f"OpenAI error: {e}")
        time.sleep(60)  # Wait for 60 seconds before retrying
        return generate_mcq_with_openai(prompt)

def generate_map():
    """Create a map using Matplotlib and Cartopy."""
    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([68, 97, 8, 37], crs=ccrs.PlateCarree())  # India extent

    # Add features to the map
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.RIVERS, edgecolor='blue')

    # Add major locations of the Maurya Kingdom
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

    # Save the map
    map_path = os.path.join(OUTPUT_DIR, "generated_map.png")
    plt.savefig(map_path)
    plt.close()
    return map_path

def generate_table():
    """Create a table using pandas."""
    data = {
        'Location': ['Pataliputra', 'Taxila', 'Ujjain', 'Kalinga'],
        'Longitude': [85.144, 72.836, 75.784, 85.833],
        'Latitude': [25.611, 33.737, 23.179, 20.936]
    }
    df = pd.DataFrame(data)
    table_path = os.path.join(OUTPUT_DIR, "generated_table.xlsx")
    df.to_excel(table_path, index=False)
    return table_path

def classify_content_type(question):
    """Use OpenAI to classify the content type."""
    prompt = f"Classify the following question into one of the categories: map, table, image.\n\nQuestion: {question}\n\nContent type:"
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        content_type = response.choices[0].message.content.strip().lower()
        return content_type
    except OpenAIError as e:
        print(f"OpenAI error: {e}")
        time.sleep(60)  # Wait for 60 seconds before retrying
        return classify_content_type(question)

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
    # Fine-tuning BERT for question answering
    model_bert, tokenizer_bert = fine_tune_bert()

    # Fine-tuning T5 for explanation generation
    model_t5, tokenizer_t5 = fine_tune_t5()

    # Test: Generate an MCQ using OpenAI
    prompt = "Generate a multiple-choice question about climate change."
    mcq = generate_mcq_with_openai(prompt)
    print(f"Generated MCQ: {mcq}")

    # Example of using BERT to verify the answer
    context = "Climate change refers to long-term changes in the temperature and weather patterns. It can be caused by human activity or natural processes."
    question = "What is climate change?"
    inputs = tokenizer_bert(question, context, return_tensors="pt")

    outputs = model_bert(**inputs)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    all_tokens = tokenizer_bert.convert_ids_to_tokens(inputs["input_ids"][0])
    start_token = torch.argmax(start_scores)
    end_token = torch.argmax(end_scores)

    bert_answer = ''.join(all_tokens[start_token:end_token+1])
    print(f"Answer from BERT: {bert_answer}")

    # Example of using T5 to generate an explanation
    input_text = f"Explain the concept of {question}."
    inputs_t5 = tokenizer_t5(input_text, return_tensors="pt", padding=True, truncation=True)

    outputs_t5 = model_t5.generate(inputs_t5["input_ids"])
    explanation = tokenizer_t5.decode(outputs_t5[0], skip_special_tokens=True)

    print(f"Explanation from T5: {explanation}")

    # Generate a prompt using OpenAI
    openai_prompt = "Generate a question related to the Maurya Kingdom in history where a map is required."
    generated_question = generate_mcq_with_openai(openai_prompt)

    # Example MCQ question related to the Maurya Kingdom
    question = "Which of the following locations was the capital of the Maurya Kingdom?"
    options = ["A) Pataliputra", "B) Taxila", "C) Ujjain", "D) Kalinga"]
    explanation = "The map shows the major locations of the Maurya Kingdom, with Pataliputra being the capital, which is option A."

    # Classify the content type based on the question
    content_type = classify_content_type(question)
    print(f"Classified content type: {content_type}")

    mcq_path, content_path = create_mcq_with_content(question, options, explanation, content_type)
    print(f"MCQ saved to {mcq_path}")
    print(f"Content saved to {content_path}")

    # Generate an answer using OpenAI
    openai_answer_prompt = f"Answer the following question: {question}"
    openai_answer = generate_mcq_with_openai(openai_answer_prompt)
    print(f"Answer from OpenAI: {openai_answer}")

    # Verify the answer using BERT
    inputs_bert = tokenizer_bert(question, return_tensors="pt")
    outputs_bert = model_bert(**inputs_bert)
    start_scores = outputs_bert.start_logits
    end_scores = outputs_bert.end_logits

    start_token = torch.argmax(start_scores)
    end_token = torch.argmax(end_scores)

    bert_answer = ''.join(tokenizer_bert.convert_ids_to_tokens(inputs_bert["input_ids"][0][start_token:end_token+1]))
    print(f"Answer from BERT: {bert_answer}")

    # Conflict resolution with weightage
    openai_weight = 1.0  # Assign weight to OpenAI answer
    bert_weight = 1.0    # Assign weight to BERT answer

    if openai_answer.lower() == bert_answer.lower():
        final_answer = openai_answer
    else:
        # If answers conflict, use weightage to determine the final answer
        print(f"Conflict detected: OpenAI Answer: {openai_answer}, BERT Answer: {bert_answer}")
        if openai_weight > bert_weight:
            final_answer = openai_answer
        elif bert_weight > openai_weight:
            final_answer = bert_answer
        else:
            final_answer = f"Conflict detected: OpenAI Answer: {openai_answer}, BERT Answer: {bert_answer}"

    print(f"Final Answer: {final_answer}")

if __name__ == "__main__":
    main()
    