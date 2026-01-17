import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler


def preprocess_ckd_data(uploaded_csv_path=None):
    """
    If uploaded_csv_path is provided:
        → use uploaded dataset
    Else:
        → use default CKD datasets
    """

    print("--- Loading Datasets ---")

    try:
        if uploaded_csv_path:
            print(f"Using uploaded dataset: {uploaded_csv_path}")
            uci_df = pd.read_csv(uploaded_csv_path)
            kaggle_df = uci_df.copy()   # federated simulation
        else:
            uci_df = pd.read_csv("../data/raw/ckd_dataset.csv")
            kaggle_df = pd.read_csv("../data/raw/kaggle_ckd.csv")

    except FileNotFoundError as e:
        print("Dataset loading error:", e)
        return

    print(f"Dataset shape: {uci_df.shape}")

    # --------------------------------------------------
    # SAME LOGIC BELOW (UNCHANGED)
    # --------------------------------------------------

    uci_df.columns = uci_df.columns.str.strip().str.lower().str.replace(' ', '_')
    kaggle_df.columns = kaggle_df.columns.str.strip().str.lower().str.replace(' ', '_')

    uci_df.rename(columns={'class': 'classification'}, inplace=True)

    if 'id' in kaggle_df.columns:
        kaggle_df.drop(columns=['id'], inplace=True)

    kaggle_df.rename(columns={'wc': 'wbcc', 'rc': 'rbcc'}, inplace=True)

    COMMON_FEATURES = [
        'age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu',
        'sc','sod','pot','hemo','pcv','wbcc','rbcc','htn','dm','cad',
        'appet','pe','ane','classification'
    ]

    uci_df = uci_df[COMMON_FEATURES]
    kaggle_df = kaggle_df[COMMON_FEATURES]

    uci_df.replace(['?', '\t?'], np.nan, inplace=True)
    kaggle_df.replace(['?', '\t?'], np.nan, inplace=True)

    num_cols = uci_df.columns.drop('classification')

    for col in num_cols:
        uci_df[col] = pd.to_numeric(uci_df[col], errors='coerce')
        kaggle_df[col] = pd.to_numeric(kaggle_df[col], errors='coerce')

    for col in num_cols:
        uci_df[col].fillna(uci_df[col].median(), inplace=True)
        kaggle_df[col].fillna(kaggle_df[col].median(), inplace=True)

    binary_map = {
        'yes': 1, 'no': 0,
        'ckd': 1, 'notckd': 0,
        'good': 1, 'poor': 0,
        'present': 1, 'notpresent': 0,
        'normal': 1, 'abnormal': 0
    }

    uci_df.replace(binary_map, inplace=True)
    kaggle_df.replace(binary_map, inplace=True)

    X_uci = uci_df.drop('classification', axis=1)
    y_uci = uci_df['classification']

    X_kaggle = kaggle_df.drop('classification', axis=1)
    y_kaggle = kaggle_df['classification']

    scaler_uci = StandardScaler()
    X_uci_scaled = scaler_uci.fit_transform(X_uci)

    scaler_kaggle = StandardScaler()
    X_kaggle_scaled = scaler_kaggle.fit_transform(X_kaggle)

    os.makedirs("../data/processed", exist_ok=True)

    uci_clean = pd.DataFrame(X_uci_scaled, columns=X_uci.columns)
    uci_clean["classification"] = y_uci.values
    uci_clean.to_csv("../data/processed/uci_clean.csv", index=False)

    kaggle_clean = pd.DataFrame(X_kaggle_scaled, columns=X_kaggle.columns)
    kaggle_clean["classification"] = y_kaggle.values
    kaggle_clean.to_csv("../data/processed/kaggle_clean.csv", index=False)

    print("Preprocessing completed successfully.")