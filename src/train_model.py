import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from utils import preprocess_data

# Load dataset
df = pd.read_csv("../data/customer_data.csv")

# Preprocess
X, y, scaler = preprocess_data(df)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)

# Save model and scaler
joblib.dump(model, "../models/churn_model.pkl")
joblib.dump(scaler, "../models/scaler.pkl")

print("✅ Model and scaler saved successfully.")