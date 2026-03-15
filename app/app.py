from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("models/xgboost_model.pkl")
scaler = joblib.load("models/scaler.pkl")

COURSE_MAP = {
    "Nursing": 33,
    "Social Work": 171,
    "Management": 8014
}

APP_MODE_MAP = {
    "General Admission": 1,
    "International Application": 2,
    "Special Admission": 5
}

PARENT_OCC_MAP = {
    "Manager": 1,
    "Professional": 2,
    "Technician": 3,
    "Service Worker": 4,
    "Unemployed": 5
}

QUAL_MAP = {
    "Basic Education": 1,
    "High School": 2,
    "Bachelor": 3,
    "Master": 4,
    "PhD": 5
}


@app.route("/")
def home():
    return render_template(
        "index.html",
        courses=COURSE_MAP.keys(),
        app_modes=APP_MODE_MAP.keys(),
        occupations=PARENT_OCC_MAP.keys(),
        quals=QUAL_MAP.keys()
    )


@app.route("/predict", methods=["POST"])
def predict():

    age = float(request.form["age"])
    admission_grade = float(request.form["admission_grade"])
    prev_grade = float(request.form["prev_grade"])

    cu1_approved = float(request.form["cu1_approved"])
    cu2_approved = float(request.form["cu2_approved"])
    cu1_grade = float(request.form["cu1_grade"])
    cu2_grade = float(request.form["cu2_grade"])

    cu1_eval = float(request.form["cu1_eval"])
    cu2_eval = float(request.form["cu2_eval"])
    cu1_enrolled = float(request.form["cu1_enrolled"])
    cu2_enrolled = float(request.form["cu2_enrolled"])

    course = COURSE_MAP[request.form["course"]]
    app_mode = APP_MODE_MAP[request.form["app_mode"]]

    mother_occ = PARENT_OCC_MAP[request.form["mother_occ"]]
    father_occ = PARENT_OCC_MAP[request.form["father_occ"]]

    mother_qual = QUAL_MAP[request.form["mother_qual"]]
    father_qual = QUAL_MAP[request.form["father_qual"]]

    tuition = int(request.form["tuition"])
    gdp = float(request.form["gdp"])
    inflation = float(request.form["inflation"])

    features = np.array([[

        cu2_approved,
        cu2_grade,
        cu1_approved,
        cu1_grade,
        tuition,
        age,
        cu2_eval,
        admission_grade,
        prev_grade,
        course,
        cu1_eval,
        father_occ,
        mother_occ,
        gdp,
        app_mode,
        cu2_enrolled,
        mother_qual,
        cu1_enrolled,
        father_qual,
        inflation

    ]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    result = "High Dropout Risk" if prediction == 1 else "Low Dropout Risk"

    return render_template(
        "index.html",
        prediction=result,
        probability=round(probability*100,2),
        courses=COURSE_MAP.keys(),
        app_modes=APP_MODE_MAP.keys(),
        occupations=PARENT_OCC_MAP.keys(),
        quals=QUAL_MAP.keys()
    )


if __name__ == "__main__":
    app.run(debug=True)