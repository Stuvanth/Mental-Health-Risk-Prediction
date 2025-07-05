from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and encoders
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
gender_encoder = joblib.load("gender_encoder.pkl")
occupation_encoder = joblib.load("occupation_encoder.pkl")
location_encoder = joblib.load("location_encoder.pkl")
stress_encoder = joblib.load("stress_encoder.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name']
    age = request.form['age']
    gender = request.form['gender']
    occupation = request.form['occupation']
    location = request.form['location']
    sleep = request.form['sleep']
    screen = request.form['screen']
    physical = request.form['physical']
    stress = request.form['stress']
    phq9 = int(request.form['phq9'])
    gad7 = int(request.form['gad7'])

    # Run your ML model here using these inputs

    # Example dummy prediction
    result = "Low Risk" if phq9 < 5 and gad7 < 5 else "Moderate Risk"

    return render_template('index.html', result=result)
if __name__ == '__main__':
    app.run(debug=True)
