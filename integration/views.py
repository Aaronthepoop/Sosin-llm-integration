from django.shortcuts import render
from django.http import JsonResponse
from llm_project.utils import generate_mcq, verify_answer, explain_answer

# View that generates MCQ, verifies answer, and gives explanation
def mcq_view(request):
    if request.method == 'GET':
        question = request.GET.get('question', 'What is the capital of France?')  # Default question if none is provided
        
        # Generate MCQ from OpenAI
        mcq = generate_mcq(question)
        
        # Assuming the answer is passed as a GET parameter
        answer = request.GET.get('answer', 'Paris')  # Default answer
        is_correct = verify_answer(answer, question)
        
        # Get the explanation from T5
        explanation = explain_answer(answer)
        
        # Return JSON response with MCQ, answer check, and explanation
        return JsonResponse({
            'question': question,
            'mcq': mcq,
            'answer': answer,
            'is_correct': is_correct,
            'explanation': explanation
        })

# in views.py
def some_function():
    from integration.views import views
    # Now you can use views

from transformers import BertForSequenceClassification, BertTokenizer
from django.conf import settings

# Load model and tokenizer from settings
model = BertForSequenceClassification.from_pretrained(settings.BERT_MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(settings.BERT_MODEL_PATH)
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the pre-trained t5-small model from Hugging Face
model_name = "t5-small"  # This is the repo_id of the model on Hugging Face

# Load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Now you can use the model and tokenizer for tasks

# integration/views.py
from django.http import HttpResponse

def index(request):
    return HttpResponse("Hello, this is the index page!")

