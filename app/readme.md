# Student Dropout Prediction Web App

This is a Flask-based web application that predicts the probability of a student dropping out using a trained machine learning model (XGBoost).

The application takes student information such as academic performance, enrollment details, family background, and economic indicators, and returns the predicted dropout risk along with the probability.

---

# Requirements

Install the required Python libraries before running the application.

pip install flask numpy pandas scikit-learn xgboost joblib

---

# Project Structure

project/

app.py  
models/
    xgboost_model.pkl
    scaler.pkl

templates/
    index.html

static/
    style.css

README.md

---

# Running the Application

Run the Flask application with the following command:

python app.py

After starting the server, open your browser and go to:

http://127.0.0.1:5000

---

# How to Use the App

1. Open the web dashboard in your browser.
2. Fill in the student information in the form.
3. Enter academic performance details.
4. Provide enrollment and family background information.
5. Click **Predict Dropout Risk**.

The system will display:

• Dropout risk classification  
• Dropout probability score

---

# Demo Buttons

The dashboard includes demo buttons that automatically fill example cases:

Low Risk Demo → fills values representing a strong student profile  
High Risk Demo → fills values representing a potential dropout case

These buttons are useful for quickly testing the model.

---

# Model Used

The prediction model used in this application is:

XGBoost Classifier

The model was trained using the **Student Dropout and Academic Success dataset** and the top 20 most important features.

---

# Dataset

Student Dropout and Academic Success Dataset

UCI Machine Learning Repository

https://archive.ics.uci.edu/ml/datasets/Student+Dropout+and+Academic+Success

---

# Notes

Make sure the following files exist before running the app:

models/xgboost_model.pkl  
models/scaler.pkl

Otherwise the application will not start.