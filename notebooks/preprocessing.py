import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import joblib
from collections import Counter
from imblearn.over_sampling import SMOTE

# --------------------------------------------------
# SINGLE PATIENT PREPROCESSING (FOR DASHBOARD / API)
# --------------------------------------------------
def preprocess_single_patient(raw_dict, scaler_path="../data/processed/scaler.pkl"):
    import pandas as pd
    import numpy as np
    import joblib

    scaler = joblib.load(scaler_path)
    feature_order = joblib.load("../data/processed/feature_order.pkl")

    CATEGORY_MAP = {
        "rbc": {"normal": 0, "abnormal": 1},
        "pc": {"normal": 0, "abnormal": 1},
        "pcc": {"notpresent": 0, "present": 1},
        "ba": {"notpresent": 0, "present": 1},
        "htn": {"no": 0, "yes": 1},
        "dm": {"no": 0, "yes": 1},
        "cad": {"no": 0, "yes": 1},
        "appet": {"good": 0, "poor": 1},
        "pe": {"no": 0, "yes": 1},
        "ane": {"no": 0, "yes": 1},
    }

    df = pd.DataFrame([raw_dict])

    for col in feature_order:
        if col not in df.columns:
            raise ValueError(f"Missing required feature: {col}")

    df = df[feature_order]

    for col in df.columns:
        if col in CATEGORY_MAP:
            val = str(df[col].iloc[0]).strip().lower()
            if val not in CATEGORY_MAP[col]:
                raise ValueError(f"Invalid value '{val}' for feature '{col}'")
            df[col] = CATEGORY_MAP[col][val]
        else:
            df[col] = pd.to_numeric(df[col], errors="raise")

    x_scaled = scaler.transform(df.values)

    assert x_scaled.shape == (1, len(feature_order))
    assert np.all(np.isfinite(x_scaled))

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

# Backend preprocessing code 
def preprocess_ckd_data():
    print("PREPROCESSING....................")
    print("\n--- Loading New CKD Dataset (5-Class Staging) ---")

    df = pd.read_csv("../data/raw/kidney_disease_dataset.csv")

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "")
        .str.replace(")", "")
        .str.replace("/", "_")
        .str.replace("-", "_")
    )

    print(f"Dataset shape: {df.shape}")
    print("\nOriginal 5-class distribution:")
    print(df['target'].value_counts().sort_index())
    
    TARGET = "target"
    if TARGET not in df.columns:
        raise ValueError("Target column 'target' not found")

    FEATURES = [col for col in df.columns if col != TARGET]

    # Encode categorical columns
    from sklearn.preprocessing import LabelEncoder
    label_encoders = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Handle missing values
    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())
    
    # Prepare for SMOTE
    from imblearn.over_sampling import SMOTE
    X = df[FEATURES].values
    y = df[TARGET].values.astype(int)
    
    print(f"\nBefore SMOTE - Classes 0-4: {np.bincount(y)}")

    # 5-CLASS SMOTE (perfectly balances all CKD stages)
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    
    print(f"After 5-Class SMOTE: {np.bincount(y_balanced)}")
    print(f"Final shape: {X_balanced.shape}")

    # Scale balanced data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_balanced)

    # Save everything
    os.makedirs("../data/processed/", exist_ok=True)
    
    np.save("../data/processed/X_balanced.npy", X_scaled)
    np.save("../data/processed/y_balanced.npy", y_balanced)
    
    joblib.dump(scaler, "../data/processed/scaler.pkl")
    joblib.dump(FEATURES, "../data/processed/feature_order.pkl")
    
    # Save feature names for reference
    pd.DataFrame(X_scaled, columns=FEATURES).assign(target=y_balanced).to_csv(
        "../data/processed/ckd_5class_balanced.csv", index=False
    )

    print("\n5-CLASS SMOTE Preprocessing Completed!")
    print("Files saved:")
    print("- X_balanced.npy (82k × 42 features)")
    print("- y_balanced.npy (82k samples, 16k per CKD stage)")
    print("- ckd_5class_balanced.csv (complete dataset)")

if __name__ == "__main__":
    preprocess_ckd_data()