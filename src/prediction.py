import joblib
import pandas as pd

RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}

def load_model(model_path="models/flood_model.pkl"):
    return joblib.load(model_path)

def load_features(feature_path="models/features.pkl"):
    return joblib.load(feature_path)

def predict_risk(model, features, values):
    X = pd.DataFrame([values], columns=features)
    prediction = int(model.predict(X)[0])
    probability = model.predict_proba(X)[0].tolist() if hasattr(model, "predict_proba") else None
    return prediction, RISK_LABELS.get(prediction, "Unknown"), probability
