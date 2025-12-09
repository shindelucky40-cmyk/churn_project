import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from logger import get_logger   # FIXED: no relative import

logger = get_logger("model_utils")

# Base paths (assumes model_utils.py is in project root)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "best_churn_model.pkl")
OHE_PATH = os.path.join(MODELS_DIR, "ohe_encoder.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_names.pkl")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

# Columns used in training
NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

# Lazy-loaded models
_model = None
_ohe = None
_scaler = None
_feature_names = None
_label_encoder = None


def load_artifacts() -> Tuple:
    global _model, _ohe, _scaler, _feature_names, _label_encoder
    try:
        if _model is None:
            logger.info(f"Loading model from {MODEL_PATH}")
            _model = joblib.load(MODEL_PATH)
        if _ohe is None:
            logger.info(f"Loading OHE from {OHE_PATH}")
            _ohe = joblib.load(OHE_PATH)
        if _scaler is None:
            logger.info(f"Loading scaler from {SCALER_PATH}")
            _scaler = joblib.load(SCALER_PATH)
        if _feature_names is None:
            logger.info(f"Loading feature names from {FEATURES_PATH}")
            _feature_names = joblib.load(FEATURES_PATH)
        if _label_encoder is None:
            logger.info(f"Loading label encoder from {LABEL_ENCODER_PATH}")
            _label_encoder = joblib.load(LABEL_ENCODER_PATH)
    except Exception as e:
        logger.exception("Error loading artifacts")
        raise e
    return _model, _ohe, _scaler, _feature_names, _label_encoder


def preprocess_single(payload: Dict[str, Any]) -> np.ndarray:
    _, ohe, scaler, feature_names, _ = load_artifacts()

    df = pd.DataFrame([payload])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="ignore").fillna(0.0)
    df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="ignore").fillna(0).astype(int)

    X_num = df[NUMERIC_COLS].astype(float).to_numpy()
    X_num_scaled = scaler.transform(X_num)

    X_cat = df[CATEGORICAL_COLS].astype(str)
    X_cat_enc = ohe.transform(X_cat)

    return np.hstack([X_num_scaled, X_cat_enc])


def predict_single(payload: Dict[str, Any]) -> Dict[str, Any]:
    model, _, _, _, label_encoder = load_artifacts()
    X = preprocess_single(payload)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0].tolist()
        pred_idx = int(np.argmax(probs))
        pred_prob = float(probs[pred_idx])
    else:
        preds = model.predict(X)[0]
        pred_idx = int(preds)
        pred_prob = None
        probs = None

    try:
        pred_label = label_encoder.inverse_transform([pred_idx])[0]
    except Exception:
        pred_label = str(pred_idx)

    return {
        "prediction": pred_idx,
        "prediction_label": pred_label,
        "probability_of_prediction": pred_prob,
        "probabilities": probs
    }


def preprocess_batch(payloads: List[Dict[str, Any]]) -> np.ndarray:
    _, ohe, scaler, _, _ = load_artifacts()
    df = pd.DataFrame(payloads)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="ignore").fillna(0.0)
    df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="ignore").fillna(0).astype(int)

    X_num = df[NUMERIC_COLS].astype(float).to_numpy()
    X_num_scaled = scaler.transform(X_num)

    X_cat = df[CATEGORICAL_COLS].astype(str)
    X_cat_enc = ohe.transform(X_cat)

    return np.hstack([X_num_scaled, X_cat_enc])


def predict_batch(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    model, _, _, _, label_encoder = load_artifacts()
    X = preprocess_batch(payloads)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X).tolist()
        preds = model.predict(X).tolist()
    else:
        preds = model.predict(X).tolist()
        probs = None

    try:
        labels = label_encoder.inverse_transform(preds).tolist()
    except Exception:
        labels = [str(p) for p in preds]

    return {
        "predictions": preds,
        "prediction_labels": labels,
        "probabilities": probs
    }
