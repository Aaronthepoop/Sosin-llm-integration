from transformers import BertTokenizer, T5Tokenizer, GPT2Tokenizer

# Example: Define the tokenizer for BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Or for T5
tokenizer = T5Tokenizer.from_pretrained('t5-small')
input_text = "Sample text for tokenization."
tokenized_input = tokenizer(input_text)  # Tokenize the input text
# utils.py

from transformers import BertTokenizer  # Import the correct tokenizer

# Instantiate the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_input(input_text):
    # Ensure tokenizer is defined before use
    tokenized_input = tokenizer(input_text)
    return tokenized_input


import openai
from transformers import BertForSequenceClassification, T5ForConditionalGeneration
import torch
from django.conf import settings

# Load models (BERT and T5)
model_bert = BertForSequenceClassification.from_pretrained(settings.BERT_MODEL_PATH)
model_t5 = T5ForConditionalGeneration.from_pretrained(settings.T5_MODEL_PATH)

def generate_mcq(question):
    response = openai.Completion.create(
        model="text-davinci-003",  # Or use your specific OpenAI model
        prompt=question,
        max_tokens=150
    )
    return response.choices[0].text.strip()

def verify_answer(answer, question):
    # Logic for answer verification using BERT
    inputs = tokenizer(question, return_tensors="pt")
    outputs = model_bert(**inputs)
    prediction = torch.argmax(outputs.logits)
    return prediction

def explain_answer(answer):
    # Logic for explanation using T5
    inputs = tokenizer(answer, return_tensors="pt")
    outputs = model_t5.generate(inputs['input_ids'])
    explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return explanation
