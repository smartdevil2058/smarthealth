"""
patient_footfall_api.py
=======================
Smart Health — Patient Footfall Prediction API
Endpoint : POST /predict/footfall
Port     : 8002

Run with:
    uvicorn patient_footfall_api:app --host 0.0.0.0 --port 8002 --reload
"""

import os
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# -------------------------------------------------------
# EDIT THESE PATHS
# -------------------------------------------------------
MODEL_PATH   = "models/patient_footfall_model.pkl"
ENCODER_PATH = "models/patient_footfall_encoders.pkl"
FEATURE_PATH = "models/patient_footfall_features.pkl"
# -------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-2.5-flash")

THRESHOLD = 0.50

app = FastAPI(
    title="Patient Footfall Prediction API",
    description="Predicts high patient footfall risk with AI explanation.",
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


class FootfallInput(BaseModel):
    Patient_Count    : float
    Holiday          : int
    Weekend          : int
    Disease_Outbreak : int
    Population       : float
    Doctors          : float
    Rainfall         : float
    Date             : str

    class Config:
        json_schema_extra = {
            "example": {
                "Patient_Count"    : 95,
                "Holiday"          : 0,
                "Weekend"          : 0,
                "Disease_Outbreak" : 1,
                "Population"       : 38952,
                "Doctors"          : 2,
                "Rainfall"         : 0,
                "Date"             : "2024-07-15"
            }
        }


class FootfallOutput(BaseModel):
    prediction          : int
    probability         : float
    risk_label          : str
    explanation_english : str
    explanation_hindi   : str


def engineer_features(data: FootfallInput) -> pd.DataFrame:
    row = pd.DataFrame([data.dict()])

    row["Date"]         = pd.to_datetime(row["Date"])
    row["day_of_week"]  = row["Date"].dt.dayofweek
    row["month"]        = row["Date"].dt.month
    row["week_of_year"] = row["Date"].dt.isocalendar().week.astype(int)
    row["quarter"]      = row["Date"].dt.quarter

    row["patients_per_doctor"]   = (row["Patient_Count"] / row["Doctors"].replace(0, 1)).clip(upper=500)
    row["population_per_doctor"] = (row["Population"] / row["Doctors"].replace(0, 1))
    row["is_high_risk_day"]      = (
        (row["Holiday"] == 1) | (row["Weekend"] == 1) | (row["Disease_Outbreak"] == 1)
    ).astype(int)
    row["outbreak_on_weekday"]   = (
        (row["Disease_Outbreak"] == 1) & (row["Weekend"] == 0) & (row["Holiday"] == 0)
    ).astype(int)

    return row[features]


def get_gemini_explanation(data: FootfallInput, risk_label: str, probability: float) -> dict:
    prompt = f"""
You are a healthcare operations assistant for Indian Primary Health Centers (PHCs).

A machine learning model has analyzed patient footfall data and produced the following result:

Patient Count    : {data.Patient_Count}
Doctors On Duty  : {data.Doctors}
Population       : {data.Population}
Holiday          : {"Yes" if data.Holiday == 1 else "No"}
Weekend          : {"Yes" if data.Weekend == 1 else "No"}
Disease Outbreak : {"Yes" if data.Disease_Outbreak == 1 else "No"}
Rainfall         : {data.Rainfall} mm
Date             : {data.Date}

Prediction       : {risk_label}
Confidence       : {round(probability * 100, 1)}%

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
        "api"      : "Patient Footfall Prediction",
        "version"  : "1.0.0",
        "endpoint" : "POST /predict/footfall"
    }


@app.post("/predict/footfall", response_model=FootfallOutput)
def predict_footfall(data: FootfallInput):
    try:
        X = engineer_features(data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature engineering failed: {str(e)}")

    try:
        prob = float(model.predict_proba(X)[0][1])
        pred = int(prob >= THRESHOLD)
        risk_label = "High Footfall" if pred == 1 else "Normal Footfall"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    try:
        explanation = get_gemini_explanation(data, risk_label, prob)
    except Exception as e:
        explanation = {
            "english": "Explanation unavailable at the moment.",
            "hindi"  : "अभी स्पष्टीकरण उपलब्ध नहीं है।"
        }

    return FootfallOutput(
        prediction          = pred,
        probability         = round(prob, 4),
        risk_label          = risk_label,
        explanation_english = explanation["english"],
        explanation_hindi   = explanation["hindi"]
    )
