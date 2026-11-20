import os
import joblib
import numpy as np

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler

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
    split_data
)


MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# --------------------------------------------------
# MODEL DEFINITIONS
# --------------------------------------------------

def get_models():

    return {
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


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    classes = list(model.classes_)
    positive_index = classes.index(1)

    probabilities = model.predict_proba(X_test)[:, positive_index]

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(
            y_test, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_test, predictions, zero_division=0
        )),
        "f1": float(f1_score(
            y_test, predictions, zero_division=0
        )),
        "roc_auc": float(roc_auc_score(y_test, probabilities))
    }


# --------------------------------------------------
# MAIN TRAINING PIPELINE
# --------------------------------------------------

def main():

    print("\n===================================")
    print("     HEALTHGUARD AI MODEL TRAINING")
    print("===================================\n")

    # LOAD DATA
    df = load_data()
    print(f"Dataset shape: {df.shape}")

    X, y = prepare_data(df)

    if set(y.unique()) != {0, 1}:
        raise ValueError("Expected binary target labels 0 and 1.")

    # INDEPENDENT TRAIN / TEST SPLIT
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    # 5-FOLD STRATIFIED CROSS VALIDATION
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    models = get_models()

    results = {}
    pipelines = {}

    print("\n===================================")
    print("     CROSS-VALIDATION RESULTS")
    print("===================================")

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    for name, model in models.items():

        print(f"\nEvaluating: {name}")

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", clone(model))
        ])

        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise"
        )

        cv_auc_mean = float(np.mean(cv_results["test_roc_auc"]))
        cv_auc_std = float(np.std(cv_results["test_roc_auc"]))

        pipelines[name] = pipeline

        results[name] = {
            "cv_accuracy": float(np.mean(cv_results["test_accuracy"])),
            "cv_precision": float(np.mean(cv_results["test_precision"])),
            "cv_recall": float(np.mean(cv_results["test_recall"])),
            "cv_f1": float(np.mean(cv_results["test_f1"])),
            "cv_roc_auc": cv_auc_mean,
            "cv_roc_auc_std": cv_auc_std
        }

        print(f"CV Accuracy : {results[name]['cv_accuracy']:.4f}")
        print(f"CV Precision: {results[name]['cv_precision']:.4f}")
        print(f"CV Recall   : {results[name]['cv_recall']:.4f}")
        print(f"CV F1       : {results[name]['cv_f1']:.4f}")
        print(f"CV ROC-AUC  : {cv_auc_mean:.4f} (+/- {cv_auc_std:.4f})")

    # SELECT MODEL USING CV ONLY
    best_model_name = max(
        results,
        key=lambda name: results[name]["cv_roc_auc"]
    )

    print("\n===================================")
    print("BEST MODEL SELECTED")
    print("===================================")

    print(best_model_name)

    # FIT SCALER ONLY ON TRAINING DATA
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # TRAIN SELECTED MODEL ON FULL TRAINING SET
    best_model = clone(models[best_model_name])

    best_model.fit(
        X_train_scaled,
        y_train
    )

    # FINAL INDEPENDENT TEST EVALUATION
    print("\n===================================")
    print("INDEPENDENT TEST RESULTS")
    print("===================================")

    test_metrics = evaluate_model(
        best_model,
        X_test_scaled,
        y_test
    )

    for metric, value in test_metrics.items():

        print(f"{metric.upper():12}: {value:.4f}")

    results[best_model_name].update({
        f"test_{metric}": value
        for metric, value in test_metrics.items()
    })

    # SAVE ARTIFACTS
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
    print("FILES SAVED SUCCESSFULLY")
    print("===================================")

    print("healthguard_model.pkl")
    print("scaler.pkl")
    print("feature_names.pkl")
    print("model_results.pkl")

    print("\nMODEL TRAINING COMPLETED!")


if __name__ == "__main__":
    main()