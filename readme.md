📘 Churn Prediction API – Production Ready (FastAPI + ML Model)

A production-style machine learning API built with FastAPI, deployed on Render, and designed to predict customer churn from structured input data.
The API exposes both single prediction and batch prediction endpoints with proper schemas, validation, logging, and CORS support.

🚀 Features

FastAPI-based backend with automatic Swagger docs

/predict → Predict churn for a single customer

/predict/batch → Predict churn for multiple customers

/health → Basic health-check endpoint

CORS enabled for any frontend

Modular code structure (schemas, model utils, logging)

Pre-loaded ML model for fast predictions

Fully containerized using Docker

Deployed on Render

📂 Project Structure
churn_project/
│
├── app/
│   ├── main.py               # FastAPI app & routes
│   ├── schemas.py            # Request/response models
│   ├── model_utils.py        # Model loading & prediction
│   ├── logger.py             # Application logs
│
├── models/
│   └── churn_model.pkl       # Trained ML model
│
├── requirements.txt          # Project dependencies
├── Dockerfile                # Deployment container
└── Procfile (optional)       # Render start command

🧠 Model

The API loads a pre-trained binary classifier (churn_model.pkl).
The model predicts whether a customer will churn (1) or not churn (0) and returns:

numerical prediction

human-readable label

probability of prediction

full probability distribution (if available)

📌 API Endpoints
1. Health Check

GET /health

{
  "status": "ok"
}

2. Single Prediction

POST /predict

Request Body
{
  "tenure": 5,
  "monthly_charges": 65.4,
  "total_charges": 324.5,
  "senior_citizen": 0,
  "gender": "Male",
  "internet_service": "Fiber optic"
}

Response
{
  "prediction": 1,
  "prediction_label": "Churn",
  "probability_of_prediction": 0.82,
  "probabilities": [0.18, 0.82]
}

3. Batch Prediction

POST /predict/batch

Request Body:
{
  "customers": [
    { ...customer1 },
    { ...customer2 }
  ]
}

🐳 Docker Deployment

Build:

docker build -t churn-api .


Run:

docker run -p 8000:8000 churn-api

☁️ Render Deployment

Push project to GitHub

Create a new Web Service on Render

Set Start Command:

uvicorn app.main:app --host 0.0.0.0 --port 8000


Deploy

🧪 Testing the API

Once deployed, open:

https://your-service.onrender.com/docs


Use the interactive Swagger UI to test predictions.

🛠️ Tech Stack

FastAPI

Python 3.11

NumPy

Scikit-Learn

Uvicorn

Docker

Render Cloud