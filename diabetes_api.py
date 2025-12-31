from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

log_reg = None
dt_model = None

class DiabetesFeatures(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

@app.on_event("startup")
def load_models():
    global log_reg, dt_model
    log_reg = joblib.load("log_reg_diabetes.pkl")
    dt_model = joblib.load("dt_diabetes.pkl")

@app.get("/")
def read_root():
    return {"message": "Diabetes prediction API running"}

@app.post("/predict/logistic")
def predict_logistic(features: DiabetesFeatures):
    data = np.array([[features.Pregnancies,
                      features.Glucose,
                      features.BloodPressure,
                      features.SkinThickness,
                      features.Insulin,
                      features.BMI,
                      features.DiabetesPedigreeFunction,
                      features.Age]])
    pred = int(log_reg.predict(data)[0])
    return {"model": "logistic_regression", "diabetes": pred}

@app.post("/predict/tree")
def predict_tree(features: DiabetesFeatures):
    data = np.array([[features.Pregnancies,
                      features.Glucose,
                      features.BloodPressure,
                      features.SkinThickness,
                      features.Insulin,
                      features.BMI,
                      features.DiabetesPedigreeFunction,
                      features.Age]])
    pred = int(dt_model.predict(data)[0])
    return {"model": "decision_tree", "diabetes": pred}
