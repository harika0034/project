import numpy as np
import pickle
import pandas as pd
import os
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('rdf.pkl', 'rb'))

# Load the scaler (if available)
try:
    scaler = pickle.load(open('scale1.pkl', 'rb'))
except:
    scaler = None

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    return render_template('input.html')

@app.route('/submit', methods=['POST'])
def submit():

    # Read values from the HTML form
    input_feature = [int(x) for x in request.form.values()]

    input_feature = np.array(input_feature).reshape(1, -1)

    # Apply scaling if a scaler exists
    if scaler:
        input_feature = scaler.transform(input_feature)

    prediction = model.predict(input_feature)

    if prediction[0] == 0:
        result = "Loan will Not be Approved"
    else:
        result = "Loan will be Approved"

    return render_template("output.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
