import torch
import openai
from openai.error import OpenAIError # type: ignore
from transformers import BertForSequenceClassification, BertTokenizer, T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import load_metric
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
    print("OpenAI API connection successful.")
except OpenAIError as e:
    print(f"OpenAI API connection failed: {e}")

def fine_tune_openai():
    # Upload the dataset
    response = openai.File.create(
        file=open("path/to/your-dataset.jsonl"),
        purpose='fine-tune'
    )

    # Fine-tune the model
    fine_tune_response = openai.FineTune.create(
        training_file=response['id'],
        model="gpt-4"
    )

    print(fine_tune_response)

    # Example prompt for evaluation
    prompt = "Translate the following English text to French: 'Good night'"

    # Generate completion using the fine-tuned model
    response = openai.Completion.create(
        model="gpt-4:ft-your-fine-tuned-model-id",
        prompt=prompt,
        max_tokens=50
    )

    evaluation_text = response.choices[0].text.strip()
    print(f"Evaluation Text: {evaluation_text}")

    # Save the evaluation results to a file
    with open("evaluation_results_openai.txt", "w") as file:
        file.write(evaluation_text)

    # Provide feedback for modifying the datasets
    feedback = "The model performed well on simple translations but struggled with complex sentences. Consider adding more complex sentences to the dataset."

    # Save the feedback to a file
    with open("feedback_openai.txt", "w") as file:
        file.write(feedback)

    print("OpenAI GPT-4 evaluation results and feedback saved.")

def fine_tune_bert():
    # Load the fine-tuned model
    model = BertForSequenceClassification.from_pretrained('path/to/fine-tuned-model')
    tokenizer = BertTokenizer.from_pretrained('path/to/fine-tuned-model')

    # Define the evaluation dataset
    eval_dataset = ...

    # Define the metric
    metric = load_metric("accuracy")

    # Define the evaluation function
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = torch.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
    )

    # Create the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Evaluate the model
    eval_results = trainer.evaluate()

    # Save the evaluation results to a file
    with open("evaluation_results_bert.txt", "w") as file:
        file.write(str(eval_results))

    # Provide feedback for modifying the datasets
    feedback = "The model performed well on classification tasks but struggled with certain categories. Consider adding more examples of those categories to the dataset."

    # Save the feedback to a file
    with open("feedback_bert.txt", "w") as file:
        file.write(feedback)

    print("BERT evaluation results and feedback saved.")

def fine_tune_t5():
    # Load the fine-tuned model
    model = T5ForConditionalGeneration.from_pretrained('path/to/fine-tuned-model')
    tokenizer = T5Tokenizer.from_pretrained('path/to/fine-tuned-model')

    # Define the evaluation dataset
    eval_dataset = ...

    # Define the metric
    metric = load_metric("rouge")

    # Define the evaluation function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        return result

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
    )

    # Create the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Evaluate the model
    eval_results = trainer.evaluate()

    # Save the evaluation results to a file
    with open("evaluation_results_t5.txt", "w") as file:
        file.write(str(eval_results))

    # Provide feedback for modifying the datasets
    feedback = "The model performed well on text generation tasks but struggled with certain types of prompts. Consider adding more examples of those prompts to the dataset."

    # Save the feedback to a file
    with open("feedback_t5.txt", "w") as file:
        file.write(feedback)

    print("T5 evaluation results and feedback saved.")