import os
import io
import pickle
import pandas as pd
from typing import Optional, List
from fastapi import FastAPI, HTTPException, UploadFile, File
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


class BatchResultRow(BaseModel):
    row_id               : int
    medicine_id          : Optional[str] = None
    prediction           : int
    risk_label           : str
    probability          : float
    explanation_english  : str


class BatchOutput(BaseModel):
    total_rows    : int
    high_risk     : int
    normal        : int
    risk_rate_pct : float
    results       : List[BatchResultRow]


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


REQUIRED_STOCKOUT_COLUMNS = [
    "Medicine_ID", "Current_Stock", "Received_Stock", "Issued_Stock",
    "Supplier_Lead_Time", "Pending_Order", "Disease_Season", "Festival", "Date"
]


def read_uploaded_table(file: UploadFile, contents: bytes) -> pd.DataFrame:
    name = (file.filename or "").lower()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(contents))
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(contents))
    raise HTTPException(status_code=422, detail="Unsupported file type. Please upload a .csv or .xlsx file.")


@app.post("/upload/stockout", response_model=BatchOutput)
async def upload_stockout(file: UploadFile = File(...)):
    contents = await file.read()

    try:
        df = read_uploaded_table(file, contents)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read file: {str(e)}")

    missing = [c for c in REQUIRED_STOCKOUT_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required columns: {', '.join(missing)}")

    results = []
    high_risk_count = 0

    for idx, row in df.iterrows():
        medicine_id = str(row["Medicine_ID"]) if pd.notna(row["Medicine_ID"]) else f"ROW-{idx+1}"
        try:
            data = StockoutInput(
                Medicine_ID        = medicine_id,
                Current_Stock      = float(row["Current_Stock"]),
                Received_Stock     = float(row["Received_Stock"]),
                Issued_Stock       = float(row["Issued_Stock"]),
                Supplier_Lead_Time = float(row["Supplier_Lead_Time"]),
                Pending_Order      = int(row["Pending_Order"]),
                Disease_Season     = str(row["Disease_Season"]) if pd.notna(row["Disease_Season"]) else "None",
                Festival           = str(row["Festival"]) if pd.notna(row["Festival"]) else "None",
                Date               = str(row["Date"]),
            )
            X = engineer_features(data)
            prob = float(model.predict_proba(X)[0][1])
            pred = int(prob >= THRESHOLD)
            risk_label = "Stockout Risk" if pred == 1 else "Safe"
            if pred == 1:
                high_risk_count += 1

            results.append(BatchResultRow(
                row_id              = idx + 1,
                medicine_id         = medicine_id,
                prediction          = pred,
                risk_label          = risk_label,
                probability         = round(prob, 4),
                explanation_english = f"{risk_label} predicted with {round(prob * 100, 1)}% confidence."
            ))
        except Exception as e:
            results.append(BatchResultRow(
                row_id              = idx + 1,
                medicine_id         = medicine_id,
                prediction          = 0,
                risk_label          = "Error",
                probability         = 0.0,
                explanation_english = f"Row could not be processed: {str(e)}"
            ))

    total = len(results)
    normal = total - high_risk_count
    risk_rate_pct = round((high_risk_count / total * 100), 1) if total else 0.0

    return BatchOutput(
        total_rows    = total,
        high_risk     = high_risk_count,
        normal        = normal,
        risk_rate_pct = risk_rate_pct,
        results       = results
    )
