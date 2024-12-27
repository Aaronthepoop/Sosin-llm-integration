from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import os
from your_module import load_and_preprocess_bert_dataset, load_and_preprocess_t5_dataset, fine_tune_bert, fine_tune_t5, generate_mcq_with_openai

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'json', 'csv', 'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            if 'bert' in filename.lower():
                if filename.endswith('.csv'):
                    tokenized_datasets = load_and_preprocess_bert_dataset_from_csv(filepath)
                elif filename.endswith('.txt'):
                    tokenized_datasets = load_and_preprocess_bert_dataset_from_txt(filepath)
                else:
                    tokenized_datasets = load_and_preprocess_bert_dataset(filepath)
                fine_tune_bert(tokenized_datasets)
            elif 't5' in filename.lower():
                if filename.endswith('.csv'):
                    tokenized_datasets = load_and_preprocess_t5_dataset_from_csv(filepath)
                elif filename.endswith('.txt'):
                    tokenized_datasets = load_and_preprocess_t5_dataset_from_txt(filepath)
                else:
                    tokenized_datasets = load_and_preprocess_t5_dataset(filepath)
                fine_tune_t5(tokenized_datasets)
            elif 'openai' in filename.lower():
                generate_mcq_with_openai(filepath)

            return redirect(url_for('upload_file', filename=filename))
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)