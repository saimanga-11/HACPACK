from fastapi import FastAPI
from pydantic import BaseModel

# Import models
from .models.prophet_model import make_prediction as prophet_predict
from .models.random_forest import make_prediction as rf_predict

# ----------- App Initialization -----------
app = FastAPI(title="INGRES Chatbot Backend")

# ----------- Request Models -----------
class ChatRequest(BaseModel):
    query: str
    lang: str = "English"   # default language

class ForecastRequest(BaseModel):
    district: str
    target_column: str = "Recharge_MCM"  # Prophet target column
    periods: int = 12       # months/years to forecast

class RFRequest(BaseModel):
    Rainfall_mm: int
    WaterLevel_m: int

# ----------- Routes -----------

@app.get("/")
def root():
    return {"message": "INGRES Chatbot Backend is running 🚀"}

@app.post("/chat")
def chat(req: ChatRequest):
    # placeholder: later call RAG pipeline + GPT-4
    return {
        "answer": f"Echo: {req.query} (in {req.lang})",
        "sources": []
    }

@app.post("/predict/forecast")
def forecast(req: ForecastRequest):
    # Call Prophet model
    forecast_result = prophet_predict({
        "district": req.district,
        "periods": req.periods
    })
    return {
        "district": req.district,
        "target_column": req.target_column,
        "periods": req.periods,
        "forecast": forecast_result
    }

@app.post("/predict/random_forest")
def rf_forecast(req: RFRequest):
    # Call Random Forest model
    forecast_result = rf_predict({
        "Rainfall_mm": req.Rainfall_mm,
        "WaterLevel_m": req.WaterLevel_m
    })
    return {
        "forecast": forecast_result
    }