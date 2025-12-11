📊 Churn Prediction API 

A production-ready FastAPI service for predicting customer churn.
Includes single prediction, batch prediction, CORS support, logging, schemas, and real-time deployment using Docker + Render.

🚀 Features

✔ Model-loaded churn prediction
✔ /predict for single record
✔ /predict/batch for multiple records
✔ Pydantic validation
✔ Production logging
✔ Clean folder structure
✔ Dockerized deployment
✔ Render hosting (free tier supported)

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

Build: docker build -t churn-api .

Run: docker run -p 8000:8000 churn-api


☁️ Render Deployment

Push project to GitHub
Create a new Web Service on Render
Set Start Command:
uvicorn app.main:app --host 0.0.0.0 --port 8000

Deploy(Render)

🚀 Live API URL
🔗 https://churn-project-ekvu.onrender.com

Swagger Docs:
👉 https://churn-project-ekvu.onrender.com/docs


🛠️ Tech Stack

FastAPI
Python 3.11.11
NumPy
pandas
Scikit-Learn
Uvicorn
Docker
Render Cloud

🎯 Author

Lalit Shinde (Lucky)
B.Tech AIML | Machine Learning Engineer
✉ GitHub: 


