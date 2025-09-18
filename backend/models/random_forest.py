# models/random_forest_model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Dummy dataset for demonstration
def generate_dummy_data():
    # Let's say we predict "Recharge_MCM" based on Rainfall_mm and WaterLevel_m
    np.random.seed(42)
    n_samples = 100
    data = pd.DataFrame({
        "Rainfall_mm": np.random.randint(50, 300, size=n_samples),
        "WaterLevel_m": np.random.randint(1, 10, size=n_samples),
        "Recharge_MCM": np.random.randint(5, 50, size=n_samples)
    })
    return data

# Train the Random Forest model
def train_model():
    data = generate_dummy_data()
    X = data[["Rainfall_mm", "WaterLevel_m"]]
    y = data["Recharge_MCM"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model

# Singleton model instance
rf_model = train_model()

# Function to make predictions
def make_prediction(input_dict):
    """
    input_dict: {
        "Rainfall_mm": int,
        "WaterLevel_m": int
    }
    """
    X_new = pd.DataFrame([input_dict])
    prediction = rf_model.predict(X_new)
    return prediction.tolist()