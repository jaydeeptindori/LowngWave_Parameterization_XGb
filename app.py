from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load models
model_clear = pickle.load(open('models/clear_model.pkl', 'rb'))
model_cloudy = pickle.load(open('models/cloudy_model.pkl', 'rb'))
model_rainy = pickle.load(open('models/rainy_model.pkl', 'rb'))
model_unified = pickle.load(open('models/unified_model.pkl', 'rb'))

# Function to predict using the appropriate model based on weather condition
def predict_weather(condition, values):
    values_np = np.array(values).reshape(1, -1)  # Reshape input values for prediction
    if condition == 'clear':
        return model_clear.predict(values_np)[0]
    elif condition == 'cloudy':
        return model_cloudy.predict(values_np)[0]
    elif condition == 'rainy':
        return model_rainy.predict(values_np)[0]
    elif condition == 'unified':
        return model_unified.predict(values_np)[0]
    else:
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        conditions = ['clear', 'cloudy', 'rainy', 'unified']
        predicted_values = {}

        # Process each condition and predict
        for condition in conditions:
            values_str = request.form.get(f'values_{condition}')
            if values_str:
                values = [float(val.strip()) for val in values_str.split(',')]
                
                if len(values) != 3:
                    return "Please enter three numeric values separated by commas."

                # Predict using the selected model
                predicted_values[condition] = predict_weather(condition, values)

        # Prepare data for rendering in HTML table
        data = {
            'Condition': [cond.capitalize() for cond in predicted_values.keys()],
            'Input Values': [request.form.get(f'values_{cond}') for cond in predicted_values.keys()],
            'Predicted Value': list(predicted_values.values())
        }
        df = pd.DataFrame(data)

        return render_template('index.html', tables=[df.to_html(classes='data', index=False)])

    return render_template('index.html', tables=[])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
