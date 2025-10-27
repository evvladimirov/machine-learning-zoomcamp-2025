from fastapi import FastAPI
from fastapi import status

import pickle
app = FastAPI()

with open("pipeline_v1.bin", "rb") as f:
    pipeline = pickle.load(f)

def predict_single(client: dict) -> float:
    return pipeline.predict_proba([client])[0, 1]

@app.get("/health")
def read_root():
    return {"message": f"{status.HTTP_200_OK}"}

@app.post("/predict")
def predict(client: dict) -> dict:

    churn = predict_single(client)

    return {
        "probability": churn,
        "churning": bool(churn > 0.5),
    }