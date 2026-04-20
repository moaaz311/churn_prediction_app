# 🏦 Bank Customer Churn Prediction System

A complete **end-to-end machine learning project** that analyzes bank customer data, predicts churn, and provides an interactive dashboard using **Streamlit**.

---

## 📌 Project Overview

This project aims to help banks **identify customers likely to leave (churn)** and take proactive actions to retain them.

It includes:
- 📊 Exploratory Data Analysis (EDA)
- 🤖 Machine Learning Model (XGBoost)
- 📈 Interactive Dashboard
- 🔮 Real-time Churn Prediction App

📸 Application 
📊 Dashboard
![Dashboard](1(1).png)

❓ Business Questions
![Business Questions](1(2).png)

🔮 Predict Customer Churn
![Predict Customer Churn](1(3).png)

📊 Prediction Result
![Prediction Result](1(4).png)

💡 Insights
![Insights](1(5).png)

ℹ️ About
![About](1(6).png)
## 📂 Project Structure

```
├── BankChurners_Analysis.ipynb
├── BankChurners_model.ipynb
├── churn_model.pkl
├── churn_prediction_app.py
└── README.md
```

---

## 📊 Dataset

- **Name:** BankChurners Dataset  
- **Records:** 10,127 customers  
- **Features:** 23 attributes  
- **Target:** Attrition (Churn / Not Churn)

---

## 🔍 Exploratory Data Analysis

Main insights:
- 📉 Churn rate ≈ **16%**
- 💳 Low transaction activity strongly linked to churn
- 😴 High inactivity → higher churn probability
- 🛒 Customers with fewer products churn more

---

## 🤖 Model Development

- **Model Used:** XGBoost Classifier  

### 📈 Model Performance

| Metric     | Score   |
|------------|--------|
| Accuracy   | 96.84% |
| Precision  | 90.40% |
| Recall     | 89.85% |
| F1-Score   | 90.12% |
| ROC-AUC    | 99.19% |

---

## 💻 Streamlit Application

Features:
- 📊 Dashboard with insights
- ❓ Business questions analysis
- 🔮 Churn prediction
- 💡 Recommendations

---

## ⚙️ Installation & Setup

```bash
git clone https://github.com/your-username/bank-churn-prediction.git
cd bank-churn-prediction
pip install -r requirements.txt
streamlit run churn_prediction_app.py
```

---

## 📦 Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- xgboost
- streamlit
- plotly
- joblib

---

## 🎯 Business Value

- Reduce churn
- Improve retention
- Identify high-risk customers

---

## 👨‍💻 Author

**Moaz Asharf**
