import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")

st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("Interactive Machine Learning App")

# Load model and scaler
model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Sidebar
st.sidebar.header("Customer Information")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 18, 80, 35)
tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 3)
balance = st.sidebar.number_input("Balance", 0.0, 200000.0, 50000.0)
products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])
credit_card = st.sidebar.selectbox("Has Credit Card", [0, 1])
active_member = st.sidebar.selectbox("Is Active Member", [0, 1])
salary = st.sidebar.number_input("Estimated Salary", 10000.0, 200000.0, 60000.0)

gender_male = 1 if gender == "Male" else 0

input_data = pd.DataFrame({
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [products],
    'HasCrCard': [credit_card],
    'IsActiveMember': [active_member],
    'EstimatedSalary': [salary],
    'Gender_Male': [gender_male]
})

scaled_input = scaler.transform(input_data)

if st.sidebar.button("Predict Churn"):
    prediction = model.predict(scaled_input)
    probability = model.predict_proba(scaled_input)[0][1]

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ Customer Likely to Churn")
    else:
        st.success("✅ Customer Likely to Stay")

    st.write(f"Churn Probability: {round(probability*100,2)}%")

# -----------------------
# Data Visualizations
# -----------------------

st.header("📈 Data Insights")

df = pd.read_csv("data/customer_data.csv")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='Exited', data=df, ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("Age Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(df['Age'], bins=10, kde=True, ax=ax2)
    st.pyplot(fig2)

st.markdown("---")
st.markdown("Built with Streamlit • Scikit-Learn • Python")