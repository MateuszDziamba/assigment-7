import os
import shutil
import mlflow.sklearn
from mlflow import MlflowClient

client = MlflowClient()
versions = client.search_model_versions("name='wine-classifier'")
latest = max(int(v.version) for v in versions)
loaded = mlflow.sklearn.load_model(f"models:/wine-classifier/{latest}")
if os.path.exists("model"):
    shutil.rmtree("model")
mlflow.sklearn.save_model(loaded, "model")
print(f"Exported wine-classifier v{latest} -> ./model")