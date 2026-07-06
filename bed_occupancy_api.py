import os
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()


MODEL_PATH   = "models/bed_occupancy_model.pkl"
ENCODER_PATH = "models/bed_occupancy_encoders.pkl"
FEATURE_PATH = "models/bed_occupancy_features.pkl"



GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-2.5-flash")

THRESHOLD = 0.30

app = FastAPI(
    title="Bed Occupancy Prediction API",
    description="Predicts high bed occupancy risk with AI explanation.",
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


class BedInput(BaseModel):
    Total_Beds             : float
    Occupied_Beds          : float
    Admissions             : float
    Discharges             : float
    Average_Length_of_Stay : float
    Disease_Outbreak       : int
    Date                   : str

    class Config:
        json_schema_extra = {
            "example": {
                "Total_Beds"             : 8,
                "Occupied_Beds"          : 7,
                "Admissions"             : 3,
                "Discharges"             : 1,
                "Average_Length_of_Stay" : 4.5,
                "Disease_Outbreak"       : 1,
                "Date"                   : "2024-07-15"
            }
        }


class BedOutput(BaseModel):
    prediction          : int
    probability         : float
    risk_label          : str
    explanation_english : str
    explanation_hindi   : str


def engineer_features(data: BedInput) -> pd.DataFrame:
    row = pd.DataFrame([data.dict()])

    row["Date"]         = pd.to_datetime(row["Date"])
    row["day_of_week"]  = row["Date"].dt.dayofweek
    row["month"]        = row["Date"].dt.month
    row["week_of_year"] = row["Date"].dt.isocalendar().week.astype(int)
    row["quarter"]      = row["Date"].dt.quarter

    row["occupancy_rate"]       = (row["Occupied_Beds"] / row["Total_Beds"].replace(0, 1)).clip(upper=1.0)
    row["net_bed_change"]       = row["Admissions"] - row["Discharges"]
    row["bed_pressure"]         = (row["Admissions"] / row["Total_Beds"].replace(0, 1)).clip(upper=5.0)
    row["avg_stay_x_occupancy"] = row["Average_Length_of_Stay"] * row["occupancy_rate"]
    row["critical_occupancy"]   = (row["occupancy_rate"] >= 0.75).astype(int)

    return row[features]


def get_gemini_explanation(data: BedInput, risk_label: str, probability: float) -> dict:
    prompt = f"""
You are a healthcare operations assistant for Indian Primary Health Centers (PHCs).

A machine learning model has analyzed bed occupancy data and produced the following result:

Total Beds              : {data.Total_Beds}
Occupied Beds           : {data.Occupied_Beds}
Admissions Today        : {data.Admissions}
Discharges Today        : {data.Discharges}
Average Length of Stay  : {data.Average_Length_of_Stay} days
Disease Outbreak        : {"Yes" if data.Disease_Outbreak == 1 else "No"}
Date                    : {data.Date}
Occupancy Rate          : {round((data.Occupied_Beds / max(data.Total_Beds, 1)) * 100, 1)}%

Prediction              : {risk_label}
Confidence              : {round(probability * 100, 1)}%

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
        "api"      : "Bed Occupancy Prediction",
        "version"  : "1.0.0",
        "endpoint" : "POST /predict/bed"
    }


@app.post("/predict/bed", response_model=BedOutput)
def predict_bed(data: BedInput):
    try:
        X = engineer_features(data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature engineering failed: {str(e)}")

    try:
        prob = float(model.predict_proba(X)[0][1])
        pred = int(prob >= THRESHOLD)
        risk_label = "High Occupancy Risk" if pred == 1 else "Normal Occupancy"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    try:
        explanation = get_gemini_explanation(data, risk_label, prob)
    except Exception as e:
        explanation = {
            "english": "Explanation unavailable at the moment.",
            "hindi"  : "अभी स्पष्टीकरण उपलब्ध नहीं है।"
        }

    return BedOutput(
        prediction          = pred,
        probability         = round(prob, 4),
        risk_label          = risk_label,
        explanation_english = explanation["english"],
        explanation_hindi   = explanation["hindi"]
    )
