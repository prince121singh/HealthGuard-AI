import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


TARGET_COLUMN = "target"


def load_data(file_path="data/heart.csv"):
    """
    Load the heart disease dataset.
    """
    df = pd.read_csv(file_path)

    return df


def clean_data(df):
    """
    Basic dataset cleaning.
    """

    df = df.copy()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Convert numeric columns where possible
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Remove rows with missing values
    df = df.dropna()

    return df


def prepare_data(df):
    """
    Prepare features and target for machine learning.
    """

    df = clean_data(df)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split dataset into training and testing sets.
    """

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def scale_data(X_train, X_test):
    """
    Standardize numerical features.
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler