from fastapi import FastAPI
from fastapi import status
from pydantic import BaseModel, Field, Literal

import pickle
app = FastAPI()

class Client(BaseModel):
    lead_source: Literal["organic_search", "paid_ads", "referral", "other"]
    number_of_courses_viewed: int = Field(..., ge=0)
    annual_income: float = Field(..., ge=0.0)


with open("pipeline_v2.bin", "rb") as f:
    pipeline = pickle.load(f)

def predict_single(client: Client) -> float:
    return pipeline.predict_proba([client.model_dump()])[0, 1]

@app.get("/health")
def read_root():
    return {"message": f"{status.HTTP_200_OK}"}

@app.post("/predict")
def predict(client: Client) -> dict:


    churn = predict_single(client)

    return {
        "probability": churn,
        "churning": bool(churn > 0.5),
    }