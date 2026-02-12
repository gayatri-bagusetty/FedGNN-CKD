import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import joblib
from collections import Counter

# --------------------------------------------------
# SINGLE PATIENT PREPROCESSING (FOR DASHBOARD / API)
# --------------------------------------------------
def preprocess_single_patient(raw_dict, scaler_path="../data/processed/scaler.pkl"):

    scaler = joblib.load(scaler_path)
    feature_order = joblib.load("../data/processed/feature_order.pkl")

    df = pd.DataFrame([raw_dict])

    # Ensure all expected features exist
    for col in feature_order:
        if col not in df.columns:
            df[col] = np.nan

    df = df[feature_order]

    binary_map = {
        'yes': 1, 'no': 0,
        'ckd': 1, 'notckd': 0,
        'poor': 1, 'good': 0,
        'present': 1, 'notpresent': 0,
        'abnormal': 1, 'normal': 0,
        '\tno': 0, '\tyes': 1,
        ' yes': 1, ' \tno': 0
    }

    # Encode categorical
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(binary_map)
                .fillna(df[col])
            )

    df = df.apply(pd.to_numeric, errors="coerce")

    # Do NOT median-fill single patient
    df = df.fillna(0)

    x_scaled = scaler.transform(df)
    return x_scaled


# --------------------------------------------------
# SINGLE DATASET PREPROCESSING (FOR INCREMENTAL FL)
# --------------------------------------------------
def preprocess_uploaded_dataset(
    csv_path="../data/uploaded_local_data.csv",
    scaler_path="../data/processed/scaler.pkl"
):

    scaler = joblib.load(scaler_path)
    feature_order = joblib.load("../data/processed/feature_order.pkl")

    df = pd.read_csv(csv_path)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    TARGET = "classification"

    for col in feature_order + [TARGET]:
        if col not in df.columns:
            df[col] = np.nan

    df = df[feature_order + [TARGET]]

    df = df.applymap(lambda x: str(x).strip().lower())

    map_dict = {
        "yes": 1, "no": 0,
        "ckd": 1, "notckd": 0,
        "normal": 0, "abnormal": 1,
        "poor": 1, "good": 0,
        "present": 1, "notpresent": 0,
        "?": np.nan, "nan": np.nan, "none": np.nan
    }

    df.replace(map_dict, inplace=True)
    df = df.apply(pd.to_numeric, errors="coerce")

    # Dataset-level median fill
    df = df.fillna(df.median())
    df = df.fillna(0)

    X = df[feature_order]
    y = df[TARGET].astype(int)

    X_scaled = scaler.transform(X)
    X_scaled = np.asarray(X_scaled, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    return X_scaled, y


# --------------------------------------------------
# FULL DATASET PREPROCESSING PIPELINE
# --------------------------------------------------
def preprocess_ckd_data():

    print("\n--- Loading Datasets ---")

    uci_df = pd.read_csv("../data/raw/ckd_dataset.csv")
    kaggle_df = pd.read_csv("../data/raw/kaggle_ckd.csv")
    synthetic_df = pd.read_csv("../data/raw/synthetic_ckd.csv")

    print(f"UCI shape: {uci_df.shape}")
    print(f"Kaggle shape: {kaggle_df.shape}")
    print(f"Synthetic shape: {synthetic_df.shape}")

    for df in [uci_df, kaggle_df, synthetic_df]:
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

    uci_df.rename(columns={"class": "classification"}, inplace=True)
    synthetic_df.rename(columns={"class": "classification"}, inplace=True, errors="ignore")

    if "id" in kaggle_df.columns:
        kaggle_df.drop(columns=["id"], inplace=True)

    kaggle_df.rename(columns={"wc": "wbcc", "rc": "rbcc"}, inplace=True)

    FEATURES = [
        'age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu',
        'sc','sod','pot','hemo','pcv','wbcc','rbcc','htn','dm','cad',
        'appet','pe','ane'
    ]

    TARGET = "classification"

    num_cols = [
        'age','bp','sg','al','su','bgr','bu','sc',
        'sod','pot','hemo','pcv','wbcc','rbcc'
    ]

    cat_cols = [
        'rbc','pc','pcc','ba','htn','dm',
        'cad','appet','pe','ane'
    ]

    binary_map = {
        'yes': 1, 'no': 0,
        'ckd': 1, 'notckd': 0,
        'poor': 1, 'good': 0,
        'present': 1, 'notpresent': 0,
        'abnormal': 1, 'normal': 0
    }

    for df in [uci_df, kaggle_df, synthetic_df]:

        df.replace(['?', '\t?'], np.nan, inplace=True)

        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col].fillna(df[col].median(), inplace=True)

            # ✅ Round to 1 decimal point
            df[col] = df[col].round(1)

        df[TARGET] = (
            df[TARGET]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(binary_map)
            .fillna(0)
            .astype(int)
        )

        for col in cat_cols:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(binary_map)
                .fillna(0)
                .astype(int)
            )

    print("\n--- Saving Cleaned Raw Files ---")

    os.makedirs("../data/processed/", exist_ok=True)

    uci_df.to_csv("../data/processed/uci_clean.csv", index=False)
    kaggle_df.to_csv("../data/processed/kaggle_clean.csv", index=False)
    synthetic_df.to_csv("../data/processed/synthetic_clean.csv", index=False)

    print("\n--- Normalizing Features ---")

    scaler = StandardScaler()
    X_uci = scaler.fit_transform(uci_df[FEATURES])

    joblib.dump(scaler, "../data/processed/scaler.pkl")
    joblib.dump(FEATURES, "../data/processed/feature_order.pkl")

    print("\n--- Preprocessing Completed Successfully ---")

if __name__ == "__main__":
    preprocess_ckd_data()