from flask import Flask, render_template, jsonify, request
import joblib
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load Random Forest model
random_forest_model = joblib.load('random_forest_model.pkl')

# Load TensorFlow Keras model
keras_model = load_model('model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get real-time input from the form for 20 features
    features = [float(request.form[f'feature{i+1}']) for i in range(20)]

    # Make predictions using the models
    input_data = np.array([features])
    random_forest_prediction = random_forest_model.predict(input_data)
    keras_prediction = keras_model.predict(input_data)

    # Ensemble predictions (average in this case)
    ensemble_prediction = (random_forest_prediction + keras_prediction) / 2

    # Display the result on the webpage
    result = f'Ensemble Prediction: {ensemble_prediction[0]}'

    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
