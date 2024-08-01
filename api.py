from flask import Flask, request, jsonify, render_template, send_from_directory, session, send_file
import os
import pickle
import pandas as pd
import pdb
import json
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)
# Configure maximum file size (2 MB)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/styles.css', methods=['GET'])
def styles():
    return send_from_directory('templates', 'styles.css')

UPLOAD_FOLDER = 'uploads/'
MODEL_FOLDER = 'models/'
ALLOWED_EXTENSIONS = {'csv'}
# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models
models = {
    'clear': pickle.load(open(os.path.join(MODEL_FOLDER, 'model_clear.pkl'), 'rb')),
    'cloudy': pickle.load(open(os.path.join(MODEL_FOLDER, 'model_cloudy.pkl'), 'rb')),
    'rainy': pickle.load(open(os.path.join(MODEL_FOLDER, 'model_rainy.pkl'), 'rb')),
    'unified': pickle.load(open(os.path.join(MODEL_FOLDER, 'model_unified.pkl'), 'rb'))
}

model_filename_map = {
    'clear': 'clear.csv',
    'cloudy': 'cloudy.csv',
    'rainy': 'rainy.csv',
    'unified': 'unified.csv'
}

model_ncols = {
    'clear': 2,
    'cloudy': 3,
    'rainy': 3,
    'unified': 3
}

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    model_name = request.form.get('model_name')
    user_tag = request.form.get('user_tag')
    if not model_name:
        return jsonify({"error": "No model name provided"}), 400
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
 
    if file:
        filename = os.path.join(UPLOAD_FOLDER, user_tag+model_filename_map[model_name])
        file.save(filename)
        print('File Uploaded Successfully')
        return jsonify({"filename": model_filename_map[model_name]}), 200
    return jsonify({"error": "File upload failed"}), 500



@app.route('/run_model', methods=['POST'])
def run_model():
    try:
        selected_models = request.json.get('models', [])
        results = {}
        user_tag = request.json.get('user_tag')
        print(selected_models, user_tag)

        for model_info in selected_models:
            model_name = model_info['model_name']
            columns = model_info['columns']
            print(user_tag+model_filename_map[model_name])

            model = models.get(model_name)
            print(model.feature_names_in_)
            if model is None:
                return jsonify({"error": f"Model {model_name} not found"}), 404

            filename = os.path.join(UPLOAD_FOLDER, user_tag+model_filename_map[model_name])
            if not os.path.exists(filename):
                return jsonify({"error": f"File for model {model_name} not found"}), 404

            data = pd.read_csv(filename)
            nfeatures = model.n_features_in_
            if len(columns) != model.n_features_in_:
                return jsonify({"error": f"Incorrect No. of columns for model {model_name}. It should be {nfeatures}"}), 400

            num_columns = len(data.columns)
            if any(col < 1 or col > model.n_features_in_ for col in columns):
                return jsonify({"error": f"Columns should be between 1 and {model.n_features_in_} for Model {model_name}."}), 400
            
            
            filtered_data = data.iloc[:, [col - 1 for col in columns]]
            predictions = model.predict(filtered_data)
            results[model_name+'_T'] = filtered_data.iloc[:, 0].round(2).values.tolist()
            results[model_name] = predictions.tolist()
        # Save results to a file for later download
        results_file = os.path.join(UPLOAD_FOLDER, f'{user_tag}_longwave_radiation.txt')
        df= pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in results.items() ])).round(2)
        df.to_csv(results_file, index=False)
        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/download_predictions', methods=['GET'])
def download_predictions():
    user_tag = request.args.get('user_tag')
    file_path = os.path.join(UPLOAD_FOLDER, f'{user_tag}_longwave_radiation.txt')
    return send_file(file_path, as_attachment=True, download_name='longwave_radiation.txt', mimetype='text/plain')




@app.route('/cleanup', methods=['POST'])
def cleanup():
    try:
        user_tag = request.json.get('user_tag')
        if not user_tag:
            return jsonify({"error": "No user tag provided"}), 400
        files_to_delete = [os.path.join(UPLOAD_FOLDER, f) for f in os.listdir(UPLOAD_FOLDER) if f.startswith(user_tag)]
        for file_path in files_to_delete:
            os.remove(file_path)
        return jsonify({"message": "Temporary files cleaned up successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File size exceeds the allowed limit of 2 MB"}), 413


if __name__ == '__main__':
    app.run(debug=True)
