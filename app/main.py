from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.schemas import CustomerPayload, PredictionResponse, BatchPayload
from app.model_utils import predict_single, predict_batch
from app.logger import get_logger

logger = get_logger("api")

app = FastAPI(
    title="Churn Prediction API (Production-style)",
    version="1.0",
    description="Predict customer churn. POST single JSON to /predict or list to /predict/batch"
)

# Allow all origins (safe for Render)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK, tags=["prediction"])
def predict(payload: CustomerPayload):
    try:
        result = predict_single(payload.dict())
    except Exception as e:
        logger.exception("Error during prediction")
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "prediction": result["prediction"],
        "prediction_label": result["prediction_label"],
        "probability_of_prediction": result.get("probability_of_prediction"),
        "probabilities": result.get("probabilities")
    }

@app.post("/predict/batch", tags=["prediction"])
def predict_batch_endpoint(payload: BatchPayload):
    try:
        res = predict_batch([c.dict() for c in payload.customers])
    except Exception as e:
        logger.exception("Error during batch prediction")
        raise HTTPException(status_code=500, detail=str(e))
    return res

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
