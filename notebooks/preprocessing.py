import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def preprocess_ckd_data():
    # 1. Load All Three Datasets
    print("--- Loading Datasets ---")
    try:
        uci_df = pd.read_csv("../data/raw/ckd_dataset.csv")
        kaggle_df = pd.read_csv("../data/raw/kaggle_ckd.csv")
        synthetic_df = pd.read_csv("../data/raw/synthetic_ckd.csv")
    except FileNotFoundError as e:
        print(f"Error: Ensure your raw data files exist in ../data/raw/. Details: {e}")
        return

    print(f"UCI shape: {uci_df.shape}")
    print(f"Kaggle shape: {kaggle_df.shape}")
    print(f"Synthetic shape: {synthetic_df.shape}")

    # 2. Standardize Column Names
    for df in [uci_df, kaggle_df, synthetic_df]:
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # 3. Align Columns & Rename Target
    uci_df.rename(columns={'class': 'classification'}, inplace=True)
    if 'id' in kaggle_df.columns: kaggle_df.drop(columns=['id'], inplace=True)
    kaggle_df.rename(columns={'wc': 'wbcc', 'rc': 'rbcc'}, inplace=True)
    # Synthetic renaming (assuming it matches Kaggle/UCI format)
    synthetic_df.rename(columns={'class': 'classification'}, inplace=True, errors='ignore')

    # 4. Select COMMON FEATURES (24 Features + 1 Target)
    COMMON_FEATURES = [
        'age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu',
        'sc','sod','pot','hemo','pcv','wbcc','rbcc','htn','dm','cad',
        'appet','pe','ane','classification'
    ]

    # Ensure all required features exist across all datasets
    for df in [uci_df, kaggle_df, synthetic_df]:
        for col in COMMON_FEATURES:
            if col not in df.columns: df[col] = np.nan

    # Keep only common features
    uci_df = uci_df[COMMON_FEATURES]
    kaggle_df = kaggle_df[COMMON_FEATURES]
    synthetic_df = synthetic_df[COMMON_FEATURES]

    # 5. Handle Missing Values and Numeric Conversion
    print("--- Handling Missing Values and Data Types ---")
    num_cols = ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','pcv','wbcc','rbcc']
    cat_cols = ['rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane']
    
    binary_map = {
        'yes': 1, 'no': 0, 'ckd': 1, 'notckd': 0, 'good': 1, 'poor': 0,
        'present': 1, 'notpresent': 0, 'normal': 1, 'abnormal': 0,
        ' \tno': 0, ' yes': 1, '\tno': 0, '\tyes': 1
    }

    for df in [uci_df, kaggle_df, synthetic_df]:
        df.replace(['?', '\t?'], np.nan, inplace=True)
        
        # Numeric conversion and median imputation
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0)
        
        # Categorical cleaning and encoding
        df['classification'] = df['classification'].astype(str).str.strip().str.lower()
        df.replace(binary_map, inplace=True)
        
        for col in cat_cols:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.strip().str.lower()
            df[col].replace(binary_map, inplace=True)
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)

    # 6. Normalize Features
    print("--- Normalizing Features ---")
    # We use a single scaler fit on UCI (Hospital A) to keep a unified feature space
    scaler = StandardScaler()
    
    # Fit and transform UCI
    X_uci = uci_df.drop('classification', axis=1)
    y_uci = uci_df['classification'].astype(int)
    X_uci_scaled = scaler.fit_transform(X_uci)

    # Transform Kaggle and Synthetic using the same scaler
    X_kag = kaggle_df.drop('classification', axis=1)
    y_kag = kaggle_df['classification'].astype(int)
    X_kag_scaled = scaler.transform(X_kag)

    X_syn = synthetic_df.drop('classification', axis=1)
    y_syn = synthetic_df['classification'].astype(int)
    X_syn_scaled = scaler.transform(X_syn)

    # 7. Save Processed Datasets
    os.makedirs("../data/processed/", exist_ok=True)
    
    datasets = {
        "uci_clean.csv": (X_uci_scaled, y_uci, X_uci.columns),
        "kaggle_clean.csv": (X_kag_scaled, y_kag, X_kag.columns),
        "synthetic_clean.csv": (X_syn_scaled, y_syn, X_syn.columns)
    }

    for filename, (X, y, cols) in datasets.items():
        clean_df = pd.DataFrame(X, columns=cols)
        clean_df['classification'] = y.values
        clean_df.to_csv(f"../data/processed/{filename}", index=False)

    print("\n--- Process Completed Successfully ---")
    print(f"Files saved: {list(datasets.keys())} to ../data/processed/")

if __name__ == "__main__":
    preprocess_ckd_data()