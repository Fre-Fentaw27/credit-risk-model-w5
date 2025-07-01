from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow
from pathlib import Path

app = FastAPI()

# Configure MLflow (absolute path)
db_path = Path(__file__).parent.parent.parent / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{db_path}")

# Load model
model_name = "CreditRisk_Random_Forest"
model_stage = "Production"
model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_stage}")

# Define input schema (REPLACE WITH YOUR ACTUAL FEATURES)
class CustomerData(BaseModel):
    credit_score: float         # Example feature
    income: float               # Example feature
    loan_amount: float          # Example feature
    # Add ALL other features your model expects
    # Must match EXACTLY what was used in training

@app.post("/predict")
async def predict(data: CustomerData):
    try:
        # Convert to DataFrame with EXACT same structure as training data
        input_df = pd.DataFrame([data.dict()])
        
        # Get prediction
        probability = model.predict_proba(input_df)[0][1]  # Probability of class 1
        
        return {
            "probability": float(probability),
            "risk_category": "high" if probability > 0.5 else "low",
            "model_used": model_name
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}. Required features: {model.metadata.get_input_schema().input_names()}"
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": model_name}
