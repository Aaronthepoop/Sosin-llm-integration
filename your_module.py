import os
import openai
from datasets import load_dataset
from transformers import BertTokenizer, T5Tokenizer, Trainer, TrainingArguments, BertForQuestionAnswering, T5ForConditionalGeneration
from dotenv import load_dotenv
import pandas as pd
from datasets import Dataset

# Load environment variables from .env file
load_dotenv()

# OpenAI API Key Setup
openai.api_key = os.getenv('OPENAI_API_KEY')

def load_and_preprocess_bert_dataset(dataset_path):
    dataset = load_dataset('json', data_files=dataset_path)
    tokenizer_bert = BertTokenizer.from_pretrained("bert-large-uncased")

    def tokenize_function(examples):
        return tokenizer_bert(
            examples["question"], examples["context"], truncation=True, padding="max_length", max_length=512
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.map(
        lambda x: {"start_positions": x["answers"]["answer_start"][0], "end_positions": x["answers"]["answer_start"][0] + len(x["answers"]["text"][0])},
        batched=True,
    )

    return tokenized_datasets

def load_and_preprocess_t5_dataset(dataset_path):
    dataset = load_dataset('json', data_files=dataset_path)
    tokenizer_t5 = T5Tokenizer.from_pretrained("t5-large")

    def tokenize_function(examples):
        return tokenizer_t5(examples["input_text"], truncation=True, padding="max_length", max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    return tokenized_datasets

def fine_tune_bert(tokenized_datasets):
    model_bert = BertForQuestionAnswering.from_pretrained("bert-large-uncased")
    tokenizer_bert = BertTokenizer.from_pretrained("bert-large-uncased")

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

    trainer_bert = Trainer(
        model=model_bert,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )

    trainer_bert.train()
    model_bert.save_pretrained('./fine_tuned_bert')
    tokenizer_bert.save_pretrained('./fine_tuned_bert')

def fine_tune_t5(tokenized_datasets):
    model_t5 = T5ForConditionalGeneration.from_pretrained("t5-large")
    tokenizer_t5 = T5Tokenizer.from_pretrained("t5-large")

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

    trainer_t5 = Trainer(
        model=model_t5,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )

    trainer_t5.train()
    model_t5.save_pretrained('./fine_tuned_t5')
    tokenizer_t5.save_pretrained('./fine_tuned_t5')

def generate_mcq_with_openai(dataset_path):
    dataset = load_dataset('json', data_files=dataset_path)

    for example in dataset['train']:
        prompt = example['prompt']
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150,
            n=1,
            stop=["\n"]
        )
        mcq = response.choices[0].text.strip()
        print(f"Generated MCQ: {mcq}")

def load_and_preprocess_bert_dataset_from_csv(dataset_path):
    df = pd.read_csv(dataset_path)
    dataset = Dataset.from_pandas(df)
    tokenizer_bert = BertTokenizer.from_pretrained("bert-large-uncased")

    def tokenize_function(examples):
        return tokenizer_bert(
            examples["question"], examples["context"], truncation=True, padding="max_length", max_length=512
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.map(
        lambda x: {"start_positions": x["answers"]["answer_start"][0], "end_positions": x["answers"]["answer_start"][0] + len(x["answers"]["text"][0])},
        batched=True,
    )

    return tokenized_datasets

def load_and_preprocess_t5_dataset_from_csv(dataset_path):
    df = pd.read_csv(dataset_path)
    dataset = Dataset.from_pandas(df)
    tokenizer_t5 = T5Tokenizer.from_pretrained("t5-large")

    def tokenize_function(examples):
        return tokenizer_t5(examples["input_text"], truncation=True, padding="max_length", max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    return tokenized_datasets

def load_and_preprocess_bert_dataset_from_txt(dataset_path):
    with open(dataset_path, 'r') as file:
        lines = file.readlines()

    data = {'question': [], 'context': [], 'answers': []}
    for line in lines:
        question, context, answer = line.strip().split('\t')
        data['question'].append(question)
        data['context'].append(context)
        data['answers'].append({'text': [answer], 'answer_start': [context.find(answer)]})

    dataset = Dataset.from_dict(data)
    tokenizer_bert = BertTokenizer.from_pretrained("bert-large-uncased")

    def tokenize_function(examples):
        return tokenizer_bert(
            examples["question"], examples["context"], truncation=True, padding="max_length", max_length=512
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.map(
        lambda x: {"start_positions": x["answers"]["answer_start"][0], "end_positions": x["answers"]["answer_start"][0] + len(x["answers"]["text"][0])},
        batched=True,
    )

    return tokenized_datasets

def load_and_preprocess_t5_dataset_from_txt(dataset_path):
    with open(dataset_path, 'r') as file:
        lines = file.readlines()

    data = {'input_text': [], 'output_text': []}
    for line in lines:
        input_text, output_text = line.strip().split('\t')
        data['input_text'].append(input_text)
        data['output_text'].append(output_text)

    dataset = Dataset.from_dict(data)
    tokenizer_t5 = T5Tokenizer.from_pretrained("t5-large")

    def tokenize_function(examples):
        return tokenizer_t5(examples["input_text"], truncation=True, padding="max_length", max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    return tokenized_datasets