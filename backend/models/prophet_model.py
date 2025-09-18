# models/prophet_model.py
import pandas as pd
from prophet import Prophet

# Function to train and predict
def make_prediction(input_data):
    """
    input_data: dict with 'district' and optional future periods
    Example: {'district': 'Chennai', 'periods': 5}
    """
    # Load your dataset
    df = pd.read_csv(df = pd.read_csv(r"C:\ingres\backend\groundwater_tn_extended.csv"))  # Make sure CSV path is correct

    # Filter for selected district
    district = input_data.get("district", "Chennai")
    df = df[df['District'] == district]

    # Prepare data for Prophet
    df_prophet = df[['Year', 'Recharge_MCM']].rename(columns={'Year':'ds', 'Recharge_MCM':'y'})

    # Train Prophet model
    model = Prophet()
    model.fit(df_prophet)

    # Make future dataframe
    periods = input_data.get("periods", 5)
    future = model.make_future_dataframe(periods=periods, freq='Y')
    forecast = model.predict(future)

    # Return last predictions as a list of dicts
    return forecast[['ds','yhat']].tail(periods).to_dict(orient='records')