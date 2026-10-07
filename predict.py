"""Apply the same Arabic normalization used during training."""

import argparse, json
import joblib
from train import normalize


def predict(model_path, text):
    cleaned = normalize(text)
    if len(cleaned) <= 2:
        raise ValueError("Please provide a longer text.")
    model = joblib.load(model_path)
    return str(model.predict([cleaned])[0])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/model.joblib")
    p.add_argument("--text", required=True)
    a = p.parse_args()
    print(json.dumps({"sentiment": predict(a.model, a.text)}, ensure_ascii=False))
