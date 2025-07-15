from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = joblib.load(model_path)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get inputs from HTML form
        age = int(request.form['Age'])
        gender = int(request.form['Gender'])
        self_employed = int(request.form['self_employed'])
        family_history = int(request.form['family_history'])
        work_interfere = int(request.form['work_interfere'])

        # Arrange inputs in same order as model training
        input_data = np.array([[age, gender, self_employed, family_history, work_interfere]])

        # Make prediction
        prediction = model.predict(input_data)

        # Format the result
        if prediction[0] == 1:
            result = "You are likely to need mental health treatment."
        else:
            result = "You are unlikely to need mental health treatment."

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text="Error: " + str(e))

if __name__ == "__main__":
    app.run(debug=True)
