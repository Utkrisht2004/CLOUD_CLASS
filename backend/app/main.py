from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Bank Churn Predictor API")

# --- Make sure your frontend URL is in allow_origins ---
# (You'll add the Vercel URL here once it's deployed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://cloud-class-olive.vercel.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load the new model
try:
    # It will look for this file inside the 'app' folder
    pipeline = joblib.load("app/bank_churn_pipeline.joblib")
    print("Bank Churn Pipeline loaded successfully.")
except FileNotFoundError:
    print("ERROR: Model file 'bank_churn_pipeline.joblib' not found.")
    print("Did you run train.py and move the file to the 'app' folder?")
    pipeline = None

# 2. Define the new input data model
class BankCustomerData(BaseModel):
    credit_score: int
    age: int
    tenure: int
    balance: float
    products_number: int
    estimated_salary: float
    country: str       # 'France', 'Spain', 'Germany'
    gender: str        # 'Male', 'Female'
    active_member: int # 1 for Yes, 0 for No

# 3. Create the prediction endpoint
@app.post("/predict_churn")
async def predict_churn(data: BankCustomerData):
    if pipeline is None:
        return {"error": "Model not loaded. Please check server logs."}

    input_df = pd.DataFrame([data.dict()])
    
    # 4. Define the feature order (must match training)
    feature_order = [
        'credit_score', 'age', 'tenure', 'balance', 'products_number',
        'estimated_salary', 'country', 'gender', 'active_member'
    ]
    input_df = input_df[feature_order]

    try:
        probabilities = pipeline.predict_proba(input_df)
        churn_probability = float(probabilities[0][1]) # Convert from numpy.float

        return {
            "prediction_label": "Churn" if churn_probability > 0.5 else "No Churn",
            "churn_probability": round(churn_probability, 4)
        }
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        print(f"ERROR: {error_msg}")
        return {"error": error_msg}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Bank Churn Prediction API."}