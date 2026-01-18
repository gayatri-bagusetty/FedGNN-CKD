import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def preprocess_ckd_data():
    # 1. Load Both Datasets
    print("--- Loading Datasets ---")
    try:
        uci_df = pd.read_csv("../data/raw/ckd_dataset.csv")
        kaggle_df = pd.read_csv("../data/raw/kaggle_ckd.csv")
    except FileNotFoundError as e:
        print(f"Error: Ensure your raw data files exist in ../data/raw/. Details: {e}")
        return

    print(f"UCI shape: {uci_df.shape}")
    print(f"Kaggle shape: {kaggle_df.shape}")

    # 2. Standardize Column Names
    uci_df.columns = uci_df.columns.str.strip().str.lower().str.replace(' ', '_')
    kaggle_df.columns = kaggle_df.columns.str.strip().str.lower().str.replace(' ', '_')

    # 3. Identify & Rename Target Column
    uci_df.rename(columns={'class': 'classification'}, inplace=True)

    # 4. Align Kaggle-Specific Columns
    if 'id' in kaggle_df.columns:
        kaggle_df.drop(columns=['id'], inplace=True)

    # Rename wc/rc to match UCI naming
    kaggle_df.rename(columns={'wc': 'wbcc', 'rc': 'rbcc'}, inplace=True)

    # 5. Select COMMON FEATURES (Federated Requirement)
    COMMON_FEATURES = [
        'age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu',
        'sc','sod','pot','hemo','pcv','wbcc','rbcc','htn','dm','cad',
        'appet','pe','ane','classification'
    ]

    uci_df = uci_df[COMMON_FEATURES]
    kaggle_df = kaggle_df[COMMON_FEATURES]

    # 6. Handle Missing Values
    print("--- Handling Missing Values ---")
    uci_df.replace(['?', '\t?'], np.nan, inplace=True)
    kaggle_df.replace(['?', '\t?'], np.nan, inplace=True)

    # Convert Numeric Columns Properly
    num_cols = uci_df.columns.drop('classification')

    for col in num_cols:
        uci_df[col] = pd.to_numeric(uci_df[col], errors='coerce')
        kaggle_df[col] = pd.to_numeric(kaggle_df[col], errors='coerce')

    # Fill Missing Values with Median
    for col in num_cols:
        uci_df[col].fillna(uci_df[col].median(), inplace=True)
        kaggle_df[col].fillna(kaggle_df[col].median(), inplace=True)

    # Clean target labels
    for df in [uci_df, kaggle_df]:
        df['classification'] = (
            df['classification']
            .astype(str)
            .str.strip()
            .str.lower()
        )

    # 7. Encode Categorical Features
    binary_map = {
        'yes': 1, 'no': 0,
        'ckd': 1, 'notckd': 0,
        'good': 1, 'poor': 0,
        'present': 1, 'notpresent': 0,
        'normal': 1, 'abnormal': 0  # Added standard ARFF categorical mapping
    }

    uci_df.replace(binary_map, inplace=True)
    kaggle_df.replace(binary_map, inplace=True)

    # 8. Separate Features & Labels
    X_uci = uci_df.drop('classification', axis=1)
    y_uci = uci_df['classification']

    X_kaggle = kaggle_df.drop('classification', axis=1)
    y_kaggle = kaggle_df['classification']

    # 9. Normalize Features (Federated Principle - Independent Scaling)
    print("--- Normalizing Features ---")
    scaler_uci = StandardScaler()
    X_uci_scaled = scaler_uci.fit_transform(X_uci)

    scaler_kaggle = StandardScaler()
    X_kaggle_scaled = scaler_kaggle.fit_transform(X_kaggle)

    # 10. Save Processed Datasets
    os.makedirs("../data/processed/", exist_ok=True)

    uci_clean = pd.DataFrame(X_uci_scaled, columns=X_uci.columns)
    uci_clean['classification'] = y_uci.values
    uci_clean.to_csv("../data/processed/uci_clean.csv", index=False)

    kaggle_clean = pd.DataFrame(X_kaggle_scaled, columns=X_kaggle.columns)
    kaggle_clean['classification'] = y_kaggle.values
    kaggle_clean.to_csv("../data/processed/kaggle_clean.csv", index=False)

    print("\n--- Process Completed Successfully ---")
    print(f"Final UCI Shape: {uci_clean.shape}")
    print(f"Final Kaggle Shape: {kaggle_clean.shape}")
    print("Files saved to ../data/processed/")

if __name__ == "__main__":
    preprocess_ckd_data()