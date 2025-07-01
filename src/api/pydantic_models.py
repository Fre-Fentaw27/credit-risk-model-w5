from pydantic import BaseModel, confloat, conint
from typing import Optional, List

# --------------------------
# Request Models (Input Validation)
# --------------------------

class CustomerFeatures(BaseModel):
    """Validation model for prediction input features"""
    # Numerical features (adjust ranges as needed)
    credit_score: conint(ge=300, le=850) = 650
    income: confloat(ge=0) = 50000.0
    loan_amount: confloat(ge=0) = 10000.0
    debt_to_income: confloat(ge=0, le=1) = 0.35
    # Add all other features your model expects
    
    class Config:
        schema_extra = {
            "example": {
                "credit_score": 720,
                "income": 65000.0,
                "loan_amount": 15000.0,
                "debt_to_income": 0.28
            }
        }

# --------------------------
# Response Models (Output Validation)
# --------------------------

class RiskPrediction(BaseModel):
    """Validation model for prediction response"""
    customer_id: str
    probability: confloat(ge=0, le=1)
    risk_category: str  # Will be validated against the list below
    
    class Config:
        schema_extra = {
            "example": {
                "customer_id": "cust_12345",
                "probability": 0.34,
                "risk_category": "low"
            }
        }

class ErrorResponse(BaseModel):
    """Standard error response format"""
    error: str
    detail: Optional[str] = None
    expected_features: Optional[List[str]] = None

# --------------------------
# Helper Validators
# --------------------------

def validate_risk_category(v: str):
    if v not in ["low", "medium", "high"]:
        raise ValueError("risk_category must be 'low', 'medium' or 'high'")
    return v

# Register the validator
RiskPrediction.update_forward_refs()