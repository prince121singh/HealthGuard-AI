import os
import joblib
import pandas as pd


MODEL_PATH = "models/healthguard_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "models/feature_names.pkl"


def load_prediction_components():
    """
    Load trained model, scaler and feature names.
    """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Trained model not found. Run: "
            "python -m core.train_model"
        )

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            "Scaler not found. Train the model first."
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    feature_names = joblib.load(FEATURES_PATH)

    return model, scaler, feature_names


def predict_risk(input_data):
    """
    Generate health-risk prediction.
    """

    model, scaler, feature_names = load_prediction_components()

    # Convert input dictionary into DataFrame
    input_df = pd.DataFrame(
        [input_data],
        columns=feature_names
    )

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Probability
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(
            input_scaled
        )[0][1]

    else:
        probability = None

    return {
        "prediction": int(prediction),
        "probability": probability
    }


def get_risk_label(prediction, probability=None):
    """
    Convert model output into a readable risk label.
    """

    if prediction == 1:

        if probability is not None:

            if probability >= 0.75:
                return "Higher predicted risk"

            elif probability >= 0.50:
                return "Elevated predicted risk"

        return "Elevated predicted risk"

    return "Lower predicted risk"