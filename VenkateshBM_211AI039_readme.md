---Overview---

The project aims to demonstrate the integration of a Random Forest model and a TensorFlow Keras deep learning model into a Flask web application. The models are used to make predictions based on user-input features through a form. The Flask app provides real-time ensemble predictions and probabilities for multiple samples.



---Prerequisites---

Before using this project, ensure you have the following prerequisites installed:
Python 
Flask
TensorFlow 
Joblib 




---Usage---

Run the Flask application:



---bash---

python app.py
Access the web application in your browser at http://localhost:5000.
Fill out the form with 20 features and submit.
View the ensemble predictions and probabilities for each sample on the webpage.

Project Structure
The project structure is organized as follows:

project-root/
|-- app.py                    # Flask application
|-- random_forest_model.pkl   # Saved Random Forest model
|-- model.h5                  # Saved TensorFlow Keras model
|-- templates/                # HTML templates
|   |-- index.html
|-- static/                   # Static files (e.g., stylesheets)
|   |-- styles.css
|-- main.py/


Dependencies
The major dependencies for this project are:
Flask
TensorFlow
Joblib


