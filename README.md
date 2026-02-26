# 📊 Customer Churn Prediction Dashboard

An interactive machine learning web application that predicts customer churn using a Random Forest classification model.  
Built with a production-style pipeline including preprocessing, model persistence, and a live Streamlit dashboard.

---

# 🚀 Project Overview

Customer churn prediction is a core problem in banking, telecom, SaaS, and subscription-based businesses.  
Retaining customers is significantly cheaper than acquiring new ones, making churn modeling a high-value business task.

This project simulates a real-world ML deployment pipeline:

1. Data preprocessing
2. Feature engineering
3. Model training
4. Model persistence
5. Interactive dashboard deployment

The result is a tangible ML product — not just a notebook.

---

# 🧠 Machine Learning Approach

## Problem Type
Binary Classification  
Target variable:
- `Exited = 1` → Customer churned  
- `Exited = 0` → Customer retained  

---

## Feature Engineering

The following input variables are used:

- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary
- Gender (one-hot encoded)

### Preprocessing Steps

1. Drop non-informative ID column
2. Encode categorical variable (Gender → binary)
3. Standardize numerical features using StandardScaler
4. Split features and target

Standardization ensures features with large magnitudes (e.g., salary) do not dominate smaller ones.

---

# 🌲 Model Selection

## Random Forest Classifier

We use a Random Forest model because:

- Handles non-linear relationships
- Robust to overfitting
- Works well with tabular data
- Provides feature importance insight

Model configuration:

- 200 estimators (trees)
- Fixed random_state for reproducibility

---

# 📈 Model Evaluation

The model outputs:

- Accuracy
- Confusion Matrix
- Churn Probability

The confusion matrix evaluates:

- True Positives (correct churn predictions)
- True Negatives (correct retention predictions)
- False Positives
- False Negatives

In a real business setting, minimizing false negatives is critical because:

Missing a churn-risk customer = lost revenue.

---

# 🖥️ Interactive Dashboard

The Streamlit app provides:

- Sidebar user inputs
- Real-time churn prediction
- Probability score output
- Churn distribution visualization
- Age distribution visualization

This transforms the model into a usable decision-support tool.

---

# 🏗️ Project Architecture
