# 🩺 HealthGuard AI

### AI-Powered Health Risk Prediction & Explainability Platform

> An interactive Machine Learning application that analyzes health parameters, predicts a potential health-risk category, and explains model predictions using SHAP-based Explainable AI.

---

## 🚀 Overview

**HealthGuard AI** is an end-to-end Data Science and Machine Learning project built with Python and Streamlit.

The project combines Machine Learning, Explainable AI, data visualization, and an interactive web interface into a single platform.

### What HealthGuard AI provides

- 🩺 Health-risk prediction
- 🎯 Prediction probability
- 🧠 SHAP-based explainability
- 📊 Dataset analytics
- 🤖 Machine Learning model comparison
- 📈 Interactive visualizations
- 📄 Prediction reports
- 📜 Prediction history
- 🌓 Multiple UI themes
- 💻 Premium Streamlit interface

> ⚠️ **Educational Disclaimer:** HealthGuard AI is an educational Machine Learning project and is not a medical diagnostic tool. Model predictions should not be used to make medical decisions.

---

# ✨ Features

## 🩺 Risk Prediction Studio

Enter health-related parameters through a structured and user-friendly interface.

The application processes the input through the trained Machine Learning pipeline and provides:

- Predicted risk category
- Prediction probability
- Visual result
- Model-based interpretation
- Prediction history

---

## 🧠 Explainable AI with SHAP

HealthGuard AI doesn't only answer:

> **"What did the model predict?"**

It also attempts to answer:

> **"Which features influenced the prediction?"**

The project uses **SHAP (SHapley Additive exPlanations)** to analyze feature contributions.

The Explainability Lab displays:

- Most influential features
- Feature values
- SHAP contribution
- Relative feature importance
- Prediction explanation

---

## 🤖 Machine Learning

The project experiments with multiple classification algorithms:

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

# 📊 Model Performance

Current evaluation results from the project's train/test split:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 83.33% | 84.62% | 78.57% | 81.48% | 94.98% |
| Random Forest | 86.67% | 88.46% | 82.14% | 85.19% | 94.20% |
| SVM | 85.00% | 88.00% | 78.57% | 83.02% | 95.42% |

> **Note:** These metrics depend on the current dataset split and training configuration. They may change after retraining or modifying the dataset.

---

# 🎨 Premium User Interface

HealthGuard AI includes a modern medical-tech inspired interface.

### UI Features

- 🌌 Midnight Theme
- 🌈 Aurora Theme
- 🏥 Clinical Light Theme
- ✨ Ambient background effects
- 📊 KPI cards
- 🎯 Interactive prediction dashboard
- 📈 Plotly visualizations
- 🧭 Navigation sidebar
- 📱 Responsive layout
- 🎨 Premium medical-tech styling

---

# 📊 Dataset Intelligence

The application provides an interactive view of the dataset.

### Analytics include

- Dataset overview
- Number of records
- Number of features
- Target distribution
- Statistical summaries
- Correlation analysis
- Feature distributions
- Interactive charts

---

# 📈 Model Lab

The Model Lab provides an overview of trained model performance.

It allows users to understand how different classification algorithms perform using standard Machine Learning evaluation metrics.

---

# 🧬 Input Features

The model uses the following features:

| Feature | Description |
|---|---|
| Age | Age of the individual |
| Sex | Encoded sex value |
| Chest Pain Type | Type of chest pain |
| Resting Blood Pressure | Resting blood pressure |
| Cholesterol | Serum cholesterol |
| Fasting Blood Sugar | Fasting blood sugar indicator |
| Resting ECG | Resting electrocardiographic result |
| Maximum Heart Rate | Maximum achieved heart rate |
| Exercise Angina | Exercise-induced angina indicator |
| ST Depression | Exercise-induced ST depression |
| Slope | Slope of peak exercise ST segment |
| CA | Number of major vessels |
| Thal | Thalassemia-related encoded feature |

---

# 🔄 Machine Learning Pipeline

```text
                  User Input
                      │
                      ▼
              Data Validation
                      │
                      ▼
             Feature Processing
                      │
                      ▼
               Feature Scaling
                      │
                      ▼
              Trained ML Model
                 ┌────┴────┐
                 │         │
                 ▼         ▼
            Prediction   Probability
                 │         │
                 └────┬────┘
                      ▼
              SHAP Explainability
                      │
                      ▼
             Human-readable Result