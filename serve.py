import json
from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd

LOG_FILE = Path("predictions.jsonl")
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


def log_prediction(features, prediction, confidence):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": features,
        "prediction": prediction,
        "confidence": confidence,
    }
    with LOG_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(features: WineFeatures):
    data = features.model_dump()
    X = pd.DataFrame([list(data.values())], columns=model.feature_names_in_)
    prediction = int(model.predict(X)[0])
    confidence = round(float(model.predict_proba(X)[0].max()), 4)
    log_prediction(data, prediction, confidence)
    return {"prediction": prediction, "confidence": confidence}


@app.get("/stats")
def stats():
    if not LOG_FILE.exists():
        return {"total_predictions": 0}
    records = [json.loads(l) for l in LOG_FILE.read_text().splitlines() if l.strip()]
    if not records:
        return {"total_predictions": 0}
    class_counts = {}
    conf_sum = 0.0
    low_conf = 0
    for r in records:
        key = str(r["prediction"])
        class_counts[key] = class_counts.get(key, 0) + 1
        conf_sum += r["confidence"]
        if r["confidence"] < 0.7:
            low_conf += 1
    return {
        "total_predictions": len(records),
        "class_distribution": class_counts,
        "average_confidence": round(conf_sum / len(records), 4),
        "low_confidence_count": low_conf,
    }