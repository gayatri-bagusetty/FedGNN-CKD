import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler

# GLOBAL CONFIG
FEATURES = [
    'age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu',
    'sc','sod','pot','hemo','pcv','wbcc','rbcc','htn','dm','cad',
    'appet','pe','ane'
]

TARGET = "classification"

NUM_COLS = [
    'age','bp','sg','al','su','bgr','bu','sc',
    'sod','pot','hemo','pcv','wbcc','rbcc'
]

CAT_COLS = [
    'rbc','pc','pcc','ba','htn','dm',
    'cad','appet','pe','ane'
]

CATEGORY_MAP = {
    'yes':1,'no':0,
    'ckd':1,'notckd':0,
    'poor':1,'good':0,
    'present':1,'notpresent':0,
    'abnormal':1,'normal':0
}

# CLEAN COLUMN NAMES
def clean_columns(df):

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    if "class" in df.columns:
        df.rename(columns={"class": "classification"}, inplace=True)

    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)

    if "wc" in df.columns:
        df.rename(columns={"wc":"wbcc"}, inplace=True)

    if "rc" in df.columns:
        df.rename(columns={"rc":"rbcc"}, inplace=True)

    return df

# HANDLE MISSING VALUES
def handle_missing_values(df):
    # Replace special missing symbols
    df.replace(['?', '\t?', ''], np.nan, inplace=True)

    # NUMERIC FEATURES
    for col in NUM_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    # CATEGORICAL FEATURES
    for col in CAT_COLS:
        df[col] = df[col].astype(str).str.strip().str.lower()
        df[col] = df[col].replace("nan", np.nan)
        df[col] = df[col].fillna(df[col].mode()[0])

    # FINAL SAFETY
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df

# ENCODE CATEGORICAL FEATURES
def encode_categorical(df):

    for col in CAT_COLS:

        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(CATEGORY_MAP)
            .fillna(0)
            .astype(int)
        )

    df[TARGET] = (
        df[TARGET]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(CATEGORY_MAP)
        .fillna(0)
        .astype(int)
    )

    return df

# NORMALIZE FEATURES
def normalize_features(datasets):
    scaler = StandardScaler()
    combined = pd.concat(datasets)
    scaler.fit(combined[FEATURES])
    os.makedirs("../data/processed/", exist_ok=True)
    joblib.dump(scaler, "../data/processed/scaler.pkl")
    joblib.dump(FEATURES, "../data/processed/feature_order.pkl")
    # Print feature order
    print("\nFeature Order:")
    for i, feature in enumerate(FEATURES, 1):
        print(f"{i}. {feature}")
        
    print("\nScaler and Features saved successfully")
    return scaler

# SINGLE PATIENT PREPROCESSING (FOR DASHBOARD / API)
def preprocess_single_patient(raw_dict):
    scaler = joblib.load("../data/processed/scaler.pkl")
    feature_order = joblib.load("../data/processed/feature_order.pkl")
    df = pd.DataFrame([raw_dict])
    # ensure correct order
    df = df[feature_order]
    for col in df.columns:
        if col in CAT_COLS:
            val = str(df[col].iloc[0]).strip().lower()
            if val not in CATEGORY_MAP:
                raise ValueError(f"Invalid value {val}")
            df[col] = CATEGORY_MAP[val]
        else:
            df[col] = pd.to_numeric(df[col], errors="raise")
    print("Processed patient features:")
    print(df)
    x_scaled = scaler.transform(df)
    return x_scaled

# DATASET PREPROCESSING (FOR INCREMENTAL FL)
def preprocess_uploaded_dataset(csv_path):
    scaler = joblib.load("../data/processed/scaler.pkl")
    feature_order = joblib.load("../data/processed/feature_order.pkl")
    df = pd.read_csv(csv_path)
    df = clean_columns(df)
    for col in feature_order + [TARGET]:
        if col not in df.columns:
            df[col] = np.nan
    df = df[feature_order + [TARGET]]
    df = handle_missing_values(df)
    df = encode_categorical(df)
    # Save temporary preprocessed CSV
    temp_csv_path = os.path.join("../data/processed", "temp_preprocessed.csv")
    df.to_csv(temp_csv_path, index=False)

    return temp_csv_path

def preprocess_ckd_data():
    print("PREPROCESSING...........")
    print("--- Loading Datasets ---")

    uci_df = pd.read_csv("../data/raw/ckd_dataset.csv")
    synthetic_df = pd.read_csv("../data/raw/ckd_synthetic.csv")

    print("UCI:",uci_df.shape)
    print("Synthetic:",synthetic_df.shape)

    # cleaning
    print("Cleaning the columns in dataset.......")
    uci_df = clean_columns(uci_df)
    synthetic_df = clean_columns(synthetic_df)

    # missing values
    print("Handling missing values for each hospital data.......")
    uci_df = handle_missing_values(uci_df)
    synthetic_df = handle_missing_values(synthetic_df)

    # encoding
    print("Encoding all categorical data in each hospital data.......")
    uci_df = encode_categorical(uci_df)
    synthetic_df = encode_categorical(synthetic_df)

    print("\n--- Saving Cleaned Raw Files ---")

    os.makedirs("../data/processed/", exist_ok=True)

    uci_df.to_csv("../data/processed/uci_clean.csv",index=False)
    synthetic_df.to_csv("../data/processed/synthetic_clean.csv",index=False)

    # normalization
    print("\n--- Normalizing Features ---")
    normalize_features([uci_df,synthetic_df])
    print("\n--- Preprocessing Completed Successfully ---")

if __name__ == "__main__":
    preprocess_ckd_data()