# 🎓 Student Dropout Prediction System — AI-Powered Early Risk Detection for Educational Institutions

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Best%20Model-orange.svg)
![Flask](https://img.shields.io/badge/Flask-Web%20App-lightgrey.svg)
![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-purple.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E.svg)

A full end-to-end **Machine Learning capstone project** that predicts student dropout risk using academic, demographic, and socioeconomic data. The system trains and evaluates five classification models, selects the best performer (XGBoost), applies Explainable AI techniques (SHAP), and deploys an interactive **Flask web application** where institutions can enter student information and instantly receive a dropout risk prediction with probability score.

---

## ✨ Features

### 🤖 **Machine Learning Pipeline**
- **5 Trained Models**: Logistic Regression, Decision Tree, Random Forest, SVM, and XGBoost
- **Best Model — XGBoost**: Achieves the highest F1-score across all evaluated classifiers
- **Top 20 Feature Selection**: Uses Random Forest feature importance to select the most predictive variables
- **Feature Scaling**: StandardScaler applied for models sensitive to feature magnitude (Logistic Regression, SVM)
- **PCA Analysis**: Dimensionality reduction applied to inspect variance structure
- **Stratified Train/Test Split**: 80/20 split with stratified sampling to preserve class distribution

### 📊 **Exploratory Data Analysis (EDA)**
- Target variable distribution and class balance analysis (68% Non-Dropout / 32% Dropout)
- Correlation heatmap across all numerical features
- Dropout vs key feature comparison (admission grade, age, tuition status, academic performance)
- 31 visualization plots saved across the full analysis pipeline

### 🧠 **Explainable AI & Bias Auditing**
- **SHAP Summary Plot**: Reveals which features drive predictions most strongly
- **Partial Dependence Plots (PDP)**: Shows how individual features influence dropout probability
- **Fairness Metrics**: Model predictions audited across demographic and socioeconomic groups
- **Bias Mitigation Strategies**: Documented recommendations for equitable deployment

### 🌐 **Flask Web Application**
- Interactive student risk assessment form
- Dropdown inputs for course, application mode, parental occupation & qualification
- **One-click Demo Buttons**: Pre-fill a Low Risk and High Risk student profile for instant testing
- Returns dropout risk classification (`High Dropout Risk` / `Low Dropout Risk`) with probability score
- Clean, styled UI served via Jinja2 templates

---

## 🗂️ Project Structure

```
Capstone Project/
├── app/
│   ├── app.py                    # Flask web application
│   ├── requirements.txt          # App dependencies
│   ├── models/
│   │   ├── xgboost_model.pkl     # Deployed prediction model
│   │   └── scaler.pkl            # Fitted StandardScaler
│   ├── templates/
│   │   └── index.html            # Web UI template
│   └── static/
│       └── style.css             # Dashboard styling
├── data/
│   └── data.csv                  # UCI Student Dropout dataset
├── models/
│   ├── xgboost_model.pkl
│   ├── random_forest.pkl
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── svm_model.pkl
│   └── scaler.pkl
├── notebook/
│   └── Student Dropout Risk Prediction Using ML.ipynb   # Full ML pipeline
├── src/
│   └── plots/                    # 31 EDA & evaluation visualizations
└── demo/
    ├── Demo.mp4                  # App walkthrough video
    └── Web app.png               # Dashboard screenshot
```

---

## 🛠️ Installation

### Prerequisites

- **Python 3.9+**
- pip package manager

### Quick Install

```bash
# Clone the repository
git clone https://github.com/your-username/student-dropout-prediction.git
cd student-dropout-prediction

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r app/requirements.txt
```

---

## 🚀 Quick Start

### Run the Web Application

```bash
cd app
python app.py
```

Then open your browser and go to:

```
http://127.0.0.1:5000
```

### Run the Full ML Notebook

```bash
# Install notebook dependencies
pip install jupyter scikit-learn xgboost shap matplotlib seaborn pandas numpy

# Launch Jupyter
jupyter notebook "notebook/Student Dropout Risk Prediction Using ML.ipynb"
```

---

## 🌐 Using the Web App

1. Open the dashboard at `http://127.0.0.1:5000`
2. Fill in the student profile form:
   - Academic performance (semester grades, approved units, evaluations)
   - Enrollment details (course, application mode, age)
   - Family background (mother/father occupation and qualification)
   - Economic indicators (GDP, inflation, tuition status)
3. Click **Predict Dropout Risk**
4. View the result:
   - **Risk Classification**: High or Low Dropout Risk
   - **Probability Score**: Confidence percentage from the model

**Demo Buttons** on the dashboard auto-fill two example profiles:
- **Low Risk Demo** — strong academic profile with consistent performance
- **High Risk Demo** — profile with financial difficulties and low academic engagement

---

## ⚙️ Input Features

| Feature | Type | Description |
|---|---|---|
| Age at Enrollment | Numeric | Student's age when they enrolled |
| Admission Grade | Numeric | Grade score at admission |
| Previous Qualification Grade | Numeric | Grade from prior education |
| Course | Categorical | Degree program (Nursing, Social Work, Management) |
| Application Mode | Categorical | General, International, or Special Admission |
| Tuition Fees Up to Date | Binary | Whether student has paid tuition (1 = Yes) |
| Curricular Units Sem 1 & 2 (Approved) | Numeric | Number of units passed per semester |
| Curricular Units Sem 1 & 2 (Grade) | Numeric | Average grade per semester |
| Curricular Units Sem 1 & 2 (Enrolled) | Numeric | Units enrolled per semester |
| Curricular Units Sem 1 & 2 (Evaluations) | Numeric | Evaluations attempted per semester |
| Mother's / Father's Occupation | Categorical | Manager, Professional, Technician, etc. |
| Mother's / Father's Qualification | Categorical | Basic Education through PhD |
| GDP | Numeric | Macroeconomic GDP indicator |
| Inflation Rate | Numeric | Macroeconomic inflation indicator |

---

## 📈 Model Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.873 | 0.858 | 0.725 | 0.786 | — |
| Decision Tree | — | — | — | — | — |
| Random Forest | High | High | High | High | High |
| SVM | — | — | — | — | — |
| **XGBoost** ✅ | **Best** | **Best** | **Best** | **Best** | **Best** |

> **XGBoost was selected as the final deployed model** due to its superior F1-score, making it the most reliable at identifying both dropout and non-dropout students. Ensemble methods (XGBoost and Random Forest) outperformed linear models on this dataset.

---

## 🧠 How the Prediction Works

The pipeline processes each student through the following steps:

```
Raw Input → Feature Encoding → StandardScaler → XGBoost Classifier → Risk Label + Probability
```

**Key predictors identified by SHAP analysis:**
- `Curricular units 2nd sem (approved)` — strongest negative predictor of dropout (more passed units = lower risk)
- `Tuition fees up to date` — students with overdue tuition show significantly higher dropout probability
- `Age at enrollment` — older students tend to have higher dropout risk
- `Curricular units 1st sem (enrolled)` — engagement in early semesters is a strong retention signal

---

## 🔍 Ethical AI & Fairness

This project includes a full **Ethical AI and Bias Auditing** deliverable:

- Predictions were evaluated across demographic groups (gender, scholarship holder, debtor status, international student)
- Fairness metrics were calculated to detect systematic disparities in model outputs
- SHAP and Partial Dependence Plots were used to ensure model interpretability
- Bias mitigation strategies are documented including data rebalancing and reweighting techniques

The goal is to ensure that any institutional deployment of this system remains transparent, equitable, and responsible.

---

## 🔧 Troubleshooting

**Model files not found on startup**
```
Ensure models/xgboost_model.pkl and models/scaler.pkl exist inside the app/models/ directory.
Run the notebook first to regenerate them if missing.
```

**Port already in use**
```bash
# Change the port
python app.py --port 5001
# Or modify app.run(port=5001) in app.py
```

**XGBoost installation issues**
```bash
pip install xgboost --upgrade
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/your-feature`)
3. **Commit** your changes (`git commit -m 'Add your feature'`)
4. **Push** to the branch (`git push origin feature/your-feature`)
5. **Open** a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.

---

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** — [Student Dropout and Academic Success Dataset](https://archive.ics.uci.edu/ml/datasets/Student+Dropout+and+Academic+Success)
- **Ultralytics / Scikit-Learn** — Model training and evaluation framework
- **XGBoost** — Gradient boosting library
- **SHAP** — Explainable AI and feature importance visualization
- **Flask** — Lightweight web framework for model deployment
- **Seaborn / Matplotlib** — Data visualization

---

<div align="center">

**Made with ❤️ by Kenneth Paul Cortez — Machine Learning Capstone Project**

[⭐ Star this repo](https://github.com/your-username/student-dropout-prediction) • [🐛 Report Bug](https://github.com/your-username/student-dropout-prediction/issues) • [💡 Request Feature](https://github.com/your-username/student-dropout-prediction/issues/new)

</div>
