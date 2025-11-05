from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Customer Churn Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-project.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load the model
try:
    # Remember to use the correct path from your root!
    pipeline = joblib.load("app/churn_pipeline.joblib")
    print("Pipeline loaded successfully.")
except FileNotFoundError:
    print("ERROR: Model file 'churn_pipeline.joblib' not found.")
    pipeline = None

# -----------------------------------------------------------------
# 👇 REPLACE your old CustomerData class with this new one
# -----------------------------------------------------------------
class CustomerData(BaseModel):
    # Old features
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    gender: str
    Partner: str
    Dependents: str
    PhoneService: str
    InternetService: str
    Contract: str
    PaymentMethod: str
    
    # --- NEW FEATURES ---
    MultipleLines: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    PaperlessBilling: str # <-- Added this one too
    
    class Config:
        schema_extra = {
            "example": {
                "tenure": 12, "MonthlyCharges": 75.5, "TotalCharges": 150, "gender": "Male",
                "Partner": "Yes", "Dependents": "No", "PhoneService": "Yes",
                "InternetService": "DSL", "Contract": "Month-to-month",
                "PaymentMethod": "Electronic check", "MultipleLines": "No",
                "OnlineSecurity": "No", "OnlineBackup": "Yes",
                "DeviceProtection": "No", "TechSupport": "No",
                "StreamingTV": "No", "StreamingMovies": "No",
                "PaperlessBilling": "Yes"
            }
        }
# -----------------------------------------------------------------

@app.post("/predict_churn")
async def predict_churn(data: CustomerData):
    print(f"INCOMING DATA DICT: {data.dict()}")
    if pipeline is None:
        return {"error": "Model not loaded. Please check server logs."}

    input_df = pd.DataFrame([data.dict()])
    
    # -----------------------------------------------------------------
    # 👇 REPLACE your old feature_order list with this new one
    # -----------------------------------------------------------------
    feature_order = [
        'tenure', 'MonthlyCharges', 'TotalCharges','gender', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    # -----------------------------------------------------------------
    
    input_df = input_df[feature_order]

    try:
        probabilities = pipeline.predict_proba(input_df)
        churn_probability = float(probabilities[0][1])

        if pd.isna(churn_probability):
            churn_probability = 0.0

        return {
            "prediction_label": "Churn" if churn_probability > 0.5 else "No Churn",
            "churn_probability": round(churn_probability, 4)
        }
    except Exception as e:
        print(f"ERROR: Prediction failed: {str(e)}")
        return {"error": f"Prediction failed: {str(e)}"}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Churn Prediction API. Go to /docs to see the endpoints."}