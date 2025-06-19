from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import numpy as np
import hmac
import hashlib
import os

app = FastAPI()

# Загрузка модели
model = joblib.load("model.pkl")

# Модель входных данных
class InputData(BaseModel):
    rsi: float
    ema50: float
    ema200: float
    macd: float
    macd_signal: float
    bb_upper: float
    bb_lower: float
    bb_width: float

# Эндпоинт для предсказания
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

# Эндпоинт для подписи
@app.post("/sign")
async def create_signature(request: Request):
    body = await request.json()
    query = body.get("query", "")
    secret = os.getenv("BINGX_SECRET_KEY_DEMO")
    if not secret:
        return {"error": "Secret key not found"}

    signature = hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()
    return {"signature": signature}
