from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open(r'A:/Project/IDS/model/model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        input_features = np.array(features).reshape(1, -1)
        
        prediction = model.predict(input_features)[0]
        
        prediction_text = f"{prediction}"

        return render_template('output.html', prediction_text=prediction_text)
    
    except Exception as e:
        error_message = f"Error: {e}"
        return render_template('output.html', prediction_text=error_message)


if __name__ == '__main__':
    app.run(debug=True)