import os
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

MODEL_PATH   = "models/medicine_stockout_model.pkl"
ENCODER_PATH = "models/medicine_stockout_encoders.pkl"
FEATURE_PATH = "models/medicine_stockout_features.pkl"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-2.5-flash")

THRESHOLD = 0.50

app = FastAPI(
    title="Medicine Stockout Prediction API",
    description="Predicts medicine stockout risk with AI explanation.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

with open(MODEL_PATH,   "rb") as f: model    = pickle.load(f)
with open(ENCODER_PATH, "rb") as f: encoders = pickle.load(f)
with open(FEATURE_PATH, "rb") as f: features = pickle.load(f)


class StockoutInput(BaseModel):
    Medicine_ID        : str
    Current_Stock      : float
    Received_Stock     : float
    Issued_Stock       : float
    Supplier_Lead_Time : float
    Pending_Order      : int
    Disease_Season     : str
    Festival           : str
    Date               : str

    class Config:
        json_schema_extra = {
            "example": {
                "Medicine_ID"        : "MED-001",
                "Current_Stock"      : 15,
                "Received_Stock"     : 0,
                "Issued_Stock"       : 30,
                "Supplier_Lead_Time" : 7,
                "Pending_Order"      : 1,
                "Disease_Season"     : "Monsoon",
                "Festival"           : "None",
                "Date"               : "2024-07-15"
            }
        }


class StockoutOutput(BaseModel):
    prediction          : int
    probability         : float
    risk_label          : str
    explanation_english : str
    explanation_hindi   : str


def engineer_features(data: StockoutInput) -> pd.DataFrame:
    row = pd.DataFrame([data.dict()])

    row["Date"]         = pd.to_datetime(row["Date"])
    row["day_of_week"]  = row["Date"].dt.dayofweek
    row["month"]        = row["Date"].dt.month
    row["week_of_year"] = row["Date"].dt.isocalendar().week.astype(int)
    row["quarter"]      = row["Date"].dt.quarter

    row["Festival"] = row["Festival"].fillna("None")

    row["net_stock_change"]    = row["Received_Stock"] - row["Issued_Stock"]
    row["stock_coverage_days"] = (
        row["Current_Stock"] / row["Issued_Stock"].replace(0, 0.1)
    ).clip(upper=365)
    row["critical_stock_flag"] = (
        row["Current_Stock"] < (row["Issued_Stock"] * row["Supplier_Lead_Time"] * 3)
    ).astype(int)
    row["reorder_urgency"] = (
        row["Supplier_Lead_Time"] / row["stock_coverage_days"].replace(0, 0.1)
    ).clip(upper=10)

    for col, le in encoders.items():
        val = str(row[col].iloc[0])
        row[col] = le.transform([val]) if val in le.classes_ else -1

    return row[features]


def get_gemini_explanation(data: StockoutInput, risk_label: str, probability: float) -> dict:
    prompt = f"""
You are a healthcare supply chain assistant for Indian Primary Health Centers (PHCs).

A machine learning model has analyzed medicine stock data and produced the following result:

Medicine ID       : {data.Medicine_ID}
Current Stock     : {data.Current_Stock} units
Issued Per Day    : {data.Issued_Stock} units
Received Stock    : {data.Received_Stock} units
Supplier Lead Time: {data.Supplier_Lead_Time} days
Pending Order     : {"Yes" if data.Pending_Order == 1 else "No"}
Disease Season    : {data.Disease_Season}
Festival          : {data.Festival}
Date              : {data.Date}

Prediction        : {risk_label}
Confidence        : {round(probability * 100, 1)}%

Give a short and clear explanation of this prediction in 2 to 3 sentences.
Then give a simple recommendation for the PHC staff.

Respond in the following format exactly:

ENGLISH:
<your explanation in English here>

HINDI:
<your explanation in Hindi here>
"""
    response = gemini.generate_content(prompt)
    text = response.text.strip()

    english = ""
    hindi   = ""

    if "ENGLISH:" in text and "HINDI:" in text:
        english = text.split("ENGLISH:")[1].split("HINDI:")[0].strip()
        hindi   = text.split("HINDI:")[1].strip()
    else:
        english = text
        hindi   = "Hindi explanation unavailable."

    return {"english": english, "hindi": hindi}


@app.get("/")
def root():
    return {
        "api"      : "Medicine Stockout Prediction",
        "version"  : "1.0.0",
        "endpoint" : "POST /predict/stockout"
    }


@app.post("/predict/stockout", response_model=StockoutOutput)
def predict_stockout(data: StockoutInput):
    try:
        X = engineer_features(data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature engineering failed: {str(e)}")

    try:
        prob = float(model.predict_proba(X)[0][1])
        pred = int(prob >= THRESHOLD)
        risk_label = "Stockout Risk" if pred == 1 else "Safe"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    try:
        explanation = get_gemini_explanation(data, risk_label, prob)
    except Exception as e:
        explanation = {
            "english": "Explanation unavailable at the moment.",
            "hindi"  : "अभी स्पष्टीकरण उपलब्ध नहीं है।"
        }

    return StockoutOutput(
        prediction          = pred,
        probability         = round(prob, 4),
        risk_label          = risk_label,
        explanation_english = explanation["english"],
        explanation_hindi   = explanation["hindi"]
    )
