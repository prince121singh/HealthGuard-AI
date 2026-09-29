from core.preprocessing import load_data, prepare_data, split_data, scale_data


df = load_data()

print("\n========== DATASET ==========")
print(df.head())

print("\n========== SHAPE ==========")
print(df.shape)

print("\n========== COLUMNS ==========")
print(df.columns.tolist())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

X, y = prepare_data(df)

print("\n========== FEATURES ==========")
print(X.shape)

print("\n========== TARGET ==========")
print(y.value_counts())

X_train, X_test, y_train, y_test = split_data(X, y)

X_train_scaled, X_test_scaled, scaler = scale_data(
    X_train,
    X_test
)

print("\n========== TRAINING DATA ==========")
print(X_train_scaled.shape)

print("\n========== TEST DATA ==========")
print(X_test_scaled.shape)

print("\n✅ DATA PIPELINE WORKING!")