import os
import torch
import openai
from transformers import BertForQuestionAnswering, BertTokenizer, T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API Key Setup
openai.api_key = os.getenv('OPENAI_API_KEY')

# Function to fine-tune BERT for Question Answering
def fine_tune_bert():
    # Load dataset (use a custom dataset or the SQuAD dataset)
    dataset = load_dataset("squad")
    
    # Load pre-trained BERT model and tokenizer
    model_bert = BertForQuestionAnswering.from_pretrained("bert-large-uncased")
    tokenizer_bert = BertTokenizer.from_pretrained("bert-large-uncased")

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer_bert(
            examples["question"], examples["context"], truncation=True, padding="max_length", max_length=512
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Format the dataset for BERT (necessary for QA tasks)
    tokenized_datasets = tokenized_datasets.map(
        lambda x: {"start_positions": x["answers"]["answer_start"][0], "end_positions": x["answers"]["answer_start"][0] + len(x["answers"]["text"][0])},
        batched=True,
    )

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results_bert",
        evaluation_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    # Initialize Trainer
    trainer_bert = Trainer(
        model=model_bert,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )

    # Fine-tune BERT
    trainer_bert.train()
    model_bert.save_pretrained('./fine_tuned_bert')
    tokenizer_bert.save_pretrained('./fine_tuned_bert')

    return model_bert, tokenizer_bert


# Function to fine-tune T5 for Explanation Generation
def fine_tune_t5():
    # Load dataset (You can replace it with your custom explanation dataset)
    dataset = load_dataset("your_custom_explanation_dataset")

    # Load pre-trained T5 model and tokenizer
    model_t5 = T5ForConditionalGeneration.from_pretrained("t5-large")
    tokenizer_t5 = T5Tokenizer.from_pretrained("t5-large")

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer_t5(examples["input_text"], truncation=True, padding="max_length", max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results_t5",
        evaluation_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    # Initialize Trainer
    trainer_t5 = Trainer(
        model=model_t5,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )

    # Fine-tune T5
    trainer_t5.train()
    model_t5.save_pretrained('./fine_tuned_t5')
    tokenizer_t5.save_pretrained('./fine_tuned_t5')

    return model_t5, tokenizer_t5


# Function to generate MCQs with OpenAI API
def generate_mcq_with_openai(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",  # Or use a fine-tuned version
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=["\n"]
    )
    return response.choices[0].text.strip()


# Main function to integrate all models together
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

    print(f"Answer from BERT: {''.join(all_tokens[start_token:end_token+1])}")

    # Example of using T5 to generate an explanation
    input_text = f"Explain the concept of {question}."
    inputs_t5 = tokenizer_t5(input_text, return_tensors="pt", padding=True, truncation=True)

    outputs_t5 = model_t5.generate(inputs_t5["input_ids"])
    explanation = tokenizer_t5.decode(outputs_t5[0], skip_special_tokens=True)

    print(f"Explanation from T5: {explanation}")

if __name__ == "__main__":
    main()