
from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("model.pkl")

class InputData(BaseModel):
    rsi: float
    ema50: float
    ema200: float
    macd: float
    macd_signal: float
    bb_upper: float
    bb_lower: float
    bb_width: float

@app.post("/predict")
async def predict(data: InputData):
    features = [[
        data.rsi,
        data.ema50,
        data.ema200,
        data.macd,
        data.macd_signal,
        data.bb_upper,
        data.bb_lower,
        data.bb_width
    ]]
    probability = model.predict_proba(features)[0][1]
    prediction = int(probability >= 0.75)
    return {"prediction": prediction, "probability": round(probability, 4)}
