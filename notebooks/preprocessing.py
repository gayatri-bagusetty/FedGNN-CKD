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
        'good': 1, 'poor': 0,
        'present': 1, 'notpresent': 0,
        'normal': 1, 'abnormal': 0,
        '\tno': 0, '\tyes': 1, ' yes': 1, ' \tno': 0
    }

    # Encode categorical
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = (
                df[col].astype(str)
                .str.strip()
                .str.lower()
                .map(binary_map)
                .fillna(df[col])
            )

    # Convert to numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # Fill missing values with median
    df = df.fillna(df.median())

    # Final fallback
    df = df.fillna(0)

    # Scale
    x_scaled = scaler.transform(df)
    return x_scaled


# --------------------------------------------------
# SINGLE DATASET PREPROCESSING (FOR INCREMENTAL FL)
# --------------------------------------------------
def preprocess_uploaded_dataset(
    csv_path="../data/uploaded_local_data.csv",
    scaler_path="../data/processed/scaler.pkl"
):
    import pandas as pd
    import numpy as np
    import joblib

    scaler = joblib.load(scaler_path)
    feature_order = joblib.load("../data/processed/feature_order.pkl")

    df = pd.read_csv(csv_path)

    # Standardize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    TARGET = "classification"

    # Ensure all required columns exist
    for col in feature_order + [TARGET]:
        if col not in df.columns:
            df[col] = np.nan

    df = df[feature_order + [TARGET]]

    # 🔥 Convert EVERYTHING to lowercase strings first
    df = df.applymap(lambda x: str(x).strip().lower())

    # Mapping exactly for your dataset
    map_dict = {
        "yes": 1, "no": 0,
        "ckd": 1, "notckd": 0,
        "normal": 1, "abnormal": 0,
        "present": 1, "notpresent": 0,
        "?": np.nan, "nan": np.nan, "none": np.nan
    }

    df.replace(map_dict, inplace=True)

    # Force numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # Fill missing values
    df = df.fillna(df.median())
    df = df.fillna(0)

    X = df[feature_order]
    y = df[TARGET].astype(int)

    # Scale features
    X_scaled = scaler.transform(X)

    # Ensure numpy arrays
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

    # --------------------------------------------------
    # STANDARDIZE COLUMN NAMES
    # --------------------------------------------------
    for df in [uci_df, kaggle_df, synthetic_df]:
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

    # --------------------------------------------------
    # ALIGN TARGET & FIX DATASET-SPECIFIC ISSUES
    # --------------------------------------------------
    uci_df.rename(columns={"class": "classification"}, inplace=True)
    synthetic_df.rename(columns={"class": "classification"}, inplace=True, errors="ignore")

    if "id" in kaggle_df.columns:
        kaggle_df.drop(columns=["id"], inplace=True)

    kaggle_df.rename(columns={"wc": "wbcc", "rc": "rbcc"}, inplace=True)

    # --------------------------------------------------
    # COMMON FEATURES (24 + TARGET)
    # --------------------------------------------------
    FEATURES = [
        'age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu',
        'sc','sod','pot','hemo','pcv','wbcc','rbcc','htn','dm','cad',
        'appet','pe','ane'
    ]
    TARGET = "classification"
    ALL_COLS = FEATURES + [TARGET]

    for df in [uci_df, kaggle_df, synthetic_df]:
        for col in ALL_COLS:
            if col not in df.columns:
                df[col] = np.nan
        df = df[ALL_COLS]

    # --------------------------------------------------
    # DATA CLEANING & ENCODING
    # --------------------------------------------------
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
        'good': 1, 'poor': 0,
        'present': 1, 'notpresent': 0,
        'normal': 1, 'abnormal': 0,
        '\tno': 0, '\tyes': 1, ' yes': 1, ' \tno': 0
    }

    for df in [uci_df, kaggle_df, synthetic_df]:

        df.replace(['?', '\t?'], np.nan, inplace=True)

        # Numeric handling
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col].fillna(df[col].median(), inplace=True)

        # Target
        df[TARGET] = (
            df[TARGET]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(binary_map)
            .fillna(0)
            .astype(int)
        )

        # Categorical
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

    # --------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------
    print("\n--- Normalizing Features ---")

    X_uci = uci_df[FEATURES]
    y_uci = uci_df[TARGET]

    X_kag = kaggle_df[FEATURES]
    y_kag = kaggle_df[TARGET]

    X_syn = synthetic_df[FEATURES]
    y_syn = synthetic_df[TARGET]

    scaler = StandardScaler()
    X_uci_scaled = scaler.fit_transform(X_uci)
    X_kag_scaled = scaler.transform(X_kag)
    X_syn_scaled = scaler.transform(X_syn)

    # --------------------------------------------------
    # SAVE ARTIFACTS
    # --------------------------------------------------
    os.makedirs("../data/processed/", exist_ok=True)

    joblib.dump(scaler, "../data/processed/scaler.pkl")
    joblib.dump(FEATURES, "../data/processed/feature_order.pkl")

    print(">>> Scaler & feature order saved")

    print("UCI class distribution:", Counter(y_uci))

    # --------------------------------------------------
    # SAVE CLEAN DATASETS
    # --------------------------------------------------
    def save_clean(name, X, y):
        df = pd.DataFrame(X, columns=FEATURES)
        df[TARGET] = y.values
        df.to_csv(f"../data/processed/{name}", index=False)

    save_clean("uci_clean.csv", X_uci_scaled, y_uci)
    save_clean("kaggle_clean.csv", X_kag_scaled, y_kag)
    save_clean("synthetic_clean.csv", X_syn_scaled, y_syn)

    print("\n--- Preprocessing Completed Successfully ---")
    print("Files saved to ../data/processed/")


# --------------------------------------------------
if __name__ == "__main__":
    preprocess_ckd_data()