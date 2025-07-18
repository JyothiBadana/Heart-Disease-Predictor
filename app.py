from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form[key]) for key in request.form]
        prediction = model.predict([features])[0]
        result = "Positive (Heart Disease Detected)" if prediction == 1 else "Negative (No Heart Disease)"
        return render_template('index.html', prediction_text=f'Prediction: {result}')
    except:
        return render_template('index.html', prediction_text="Error in input format. Please check your values.")

if __name__ == '__main__':
    app.run(debug=True)
