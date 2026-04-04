# training.py
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


MODEL_PATH = Path("model.joblib")


def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names
    return X, y, target_names


def train_model():
    X, y, target_names = load_data()

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )
    model.fit(X, y)

    return model, target_names


def save_model(model, target_names, model_path: Path = MODEL_PATH):
    artifact = {
        "model": model,
        "target_names": list(target_names),
    }
    joblib.dump(artifact, model_path)


if __name__ == "__main__":
    model, target_names = train_model()
    save_model(model, target_names)
    print(f"Model saved to: {MODEL_PATH.resolve()}")
