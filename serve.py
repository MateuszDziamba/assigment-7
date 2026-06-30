from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = mlflow.sklearn.load_model("model")
    yield


app = FastAPI(title="Wine Classifier API", lifespan=lifespan)


class WineFeatures(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(features: WineFeatures):
    values = list(features.model_dump().values())
    X = pd.DataFrame([values], columns=model.feature_names_in_)
    pred = int(model.predict(X)[0])
    confidence = float(model.predict_proba(X)[0].max())
    return {"prediction": pred, "confidence": round(confidence, 4)}