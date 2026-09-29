import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from core.preprocessing import (
    load_data,
    prepare_data,
    split_data,
    scale_data
)


MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained classification model.
    """

    predictions = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
    else:
        probabilities = model.decision_function(X_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities)
    }


def main():

    print("\n===================================")
    print("     HEALTHGUARD AI MODEL TRAINING")
    print("===================================\n")

    # --------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------

    df = load_data()

    print(f"Dataset shape: {df.shape}")

    # --------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------

    X, y = prepare_data(df)

    # --------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # --------------------------------------------------
    # SCALE DATA
    # --------------------------------------------------

    X_train_scaled, X_test_scaled, scaler = scale_data(
        X_train,
        X_test
    )

    # --------------------------------------------------
    # MODELS
    # --------------------------------------------------

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=2000
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42
        ),

        "Support Vector Machine": SVC(
            probability=True,
            random_state=42
        )
    }

    results = {}

    # --------------------------------------------------
    # TRAIN MODELS
    # --------------------------------------------------

    for name, model in models.items():

        print(f"\nTraining: {name}")

        model.fit(
            X_train_scaled,
            y_train
        )

        metrics = evaluate_model(
            model,
            X_test_scaled,
            y_test
        )

        results[name] = metrics

        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")
        print(f"F1 Score : {metrics['f1']:.4f}")
        print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")

    # --------------------------------------------------
    # FIND BEST MODEL
    # --------------------------------------------------

    best_model_name = max(
        results,
        key=lambda name: results[name]["roc_auc"]
    )

    print("\n===================================")
    print("BEST MODEL")
    print("===================================")

    print(best_model_name)

    print("\nMetrics:")

    for metric, value in results[best_model_name].items():
        print(f"{metric}: {value:.4f}")

    # --------------------------------------------------
    # RETRAIN BEST MODEL
    # --------------------------------------------------

    best_model = models[best_model_name]

    best_model.fit(
        X_train_scaled,
        y_train
    )

    # --------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------

    joblib.dump(
        best_model,
        os.path.join(MODEL_DIR, "healthguard_model.pkl")
    )

    joblib.dump(
        scaler,
        os.path.join(MODEL_DIR, "scaler.pkl")
    )

    joblib.dump(
        list(X.columns),
        os.path.join(MODEL_DIR, "feature_names.pkl")
    )

    joblib.dump(
        results,
        os.path.join(MODEL_DIR, "model_results.pkl")
    )

    print("\n===================================")
    print("FILES SAVED")
    print("===================================")

    print("✓ models/healthguard_model.pkl")
    print("✓ models/scaler.pkl")
    print("✓ models/feature_names.pkl")
    print("✓ models/model_results.pkl")

    print("\n✅ MODEL TRAINING COMPLETED!")


if __name__ == "__main__":
    main()