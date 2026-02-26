import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    # Drop ID column
    df = df.drop(['CustomerID'], axis=1)

    # Encode categorical variable
    df = pd.get_dummies(df, columns=['Gender'], drop_first=True)

    # Separate features and target
    X = df.drop('Exited', axis=1)
    y = df['Exited']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler