from core.predictor import predict_risk, get_risk_label


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


result = predict_risk(sample_patient)

prediction = result["prediction"]
probability = result["probability"]

label = get_risk_label(
    prediction,
    probability
)

print("\n===================================")
print("       HEALTHGUARD AI PREDICTION")
print("===================================")

print(f"\nPrediction: {prediction}")
print(f"Risk Category: {label}")

if probability is not None:
    print(
        f"Model probability: "
        f"{probability * 100:.2f}%"
    )

print("\n✅ Prediction engine working!")