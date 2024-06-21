from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json()

    # Convert JSON data to a DataFrame
    df = pd.DataFrame([data])

    # Ensure the columns are in the correct order
    df = df[['initial payment', 'last payment', 'credit score', 'house number']]

    # Make prediction
    prediction = model.predict(df)
    result = prediction[0]

    # Return the prediction result as JSON
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
