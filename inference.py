from pathlib import Path

import joblib


MODEL_PATH = Path("model.joblib")


class IrisModelService:
    def __init__(self, model, target_names):
        self.model = model
        self.target_names = target_names

    def predict(self, features: dict) -> str:
        ordered_features = [
            [
                features["sepal_length"],
                features["sepal_width"],
                features["petal_length"],
                features["petal_width"],
            ]
        ]
        prediction_index = self.model.predict(ordered_features)[0]
        return str(self.target_names[prediction_index])


def load_model(model_path: Path = MODEL_PATH) -> IrisModelService:
    artifact = joblib.load(model_path)
    model = artifact["model"]
    target_names = artifact["target_names"]
    return IrisModelService(model=model, target_names=target_names)
