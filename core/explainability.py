import shap
import pandas as pd
import numpy as np
import joblib

from core.preprocessing import load_data


MODEL_PATH = "models/healthguard_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "models/feature_names.pkl"


def load_explainability_components():
    """
    Load the trained model, scaler and feature names.
    """

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)

    return model, scaler, feature_names


def create_background_data(feature_names):
    """
    Create a small background dataset for SHAP.
    """

    df = load_data()

    background = df[
        feature_names
    ].copy()

    # Keep SHAP computation reasonably fast
    if len(background) > 50:
        background = background.sample(
            50,
            random_state=42
        )

    return background


def create_explainer():
    """
    Create a SHAP explainer around the complete
    preprocessing + model pipeline.
    """

    model, scaler, feature_names = (
        load_explainability_components()
    )

    background = create_background_data(
        feature_names
    )

    def model_probability(data):

        data_df = pd.DataFrame(
            data,
            columns=feature_names
        )

        scaled_data = scaler.transform(
            data_df
        )

        probabilities = model.predict_proba(
            scaled_data
        )[:, 1]

        return probabilities

    explainer = shap.Explainer(
        model_probability,
        background
    )

    return (
        explainer,
        feature_names,
        background
    )


def explain_prediction(input_data):
    """
    Generate SHAP values for one prediction.
    """

    explainer, feature_names, background = (
        create_explainer()
    )

    input_df = pd.DataFrame(
        [input_data],
        columns=feature_names
    )

    shap_result = explainer(
        input_df
    )

    values = np.asarray(
        shap_result.values
    )

    # Make sure we get one value per feature
    if values.ndim == 2:

        values = values[0]

    elif values.ndim == 3:

        values = values[0, :, 0]

    feature_values = input_df.iloc[
        0
    ].values

    explanation_df = pd.DataFrame(
        {
            "feature": feature_names,
            "value": feature_values,
            "shap_value": values
        }
    )

    explanation_df[
        "absolute_shap"
    ] = explanation_df[
        "shap_value"
    ].abs()

    explanation_df = (
        explanation_df
        .sort_values(
            "absolute_shap",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return explanation_df


def get_top_features(
    explanation_df,
    top_n=8
):
    """
    Return the most influential features.
    """

    return explanation_df.head(
        top_n
    ).copy()


if __name__ == "__main__":

    sample_patient = {
        "age": 55,
        "sex": 1,
        "cp": 2,
        "trestbps": 140,
        "chol": 240,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 2,
        "ca": 0,
        "thal": 3
    }

    print("\n===================================")
    print("      HEALTHGUARD SHAP ANALYSIS")
    print("===================================\n")

    explanation = explain_prediction(
        sample_patient
    )

    print(
        explanation[
            [
                "feature",
                "value",
                "shap_value"
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n✅ SHAP explanation generated!"
    )