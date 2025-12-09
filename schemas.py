from pydantic import BaseModel, Field
from typing import Literal, Optional, List

# Numeric fields only
class NumericFields(BaseModel):
    SeniorCitizen: int = Field(..., ge=0, le=1, description="0 or 1")
    tenure: float = Field(..., ge=0, description="customer tenure in months")
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)

# Main payload for single customer
class CustomerPayload(NumericFields):
    gender: Literal["Male", "Female"]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]

# Response model for single prediction
class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    probability_of_prediction: Optional[float]
    probabilities: Optional[List[float]]

# Batch prediction request body
class BatchPayload(BaseModel):
    customers: List[CustomerPayload]
