import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from source.plot_hospital_distribution import plot_smote_comparison

# load cleaned datasets
def load_datasets():
    print("\nLoading processed datasets...")
    try:
        uci_clean = pd.read_csv("../data/processed/uci_clean.csv")
        synthetic_clean = pd.read_csv("../data/processed/synthetic_clean.csv")
    except FileNotFoundError as e:
        raise FileNotFoundError("Processed files not found. Run preprocessing.py first.") from e
    return uci_clean, synthetic_clean

# create hospital directories
def create_directories(base_path, hospitals):
    for hospital in hospitals:
        path = os.path.join(base_path, hospital)
        os.makedirs(path, exist_ok=True)

# split into train/val/test
def train_val_test_split(df):
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["classification"],
        random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["classification"],
        random_state=42
    )
    return train_df, val_df, test_df

# impute missing values
def handle_missing_values(df):
    imputer = SimpleImputer(strategy="median")
    X = df.drop("classification", axis=1)
    y = df["classification"]
    X_imputed = imputer.fit_transform(X)
    clean_df = pd.concat(
        [
            pd.DataFrame(X_imputed, columns=X.columns),
            y.reset_index(drop=True)
        ],
        axis=1
    )
    return clean_df

# apply SMOTE on training data
def apply_smote(train_df):
    X = train_df.drop("classification", axis=1)
    y = train_df["classification"]
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    balanced_df = pd.concat(
        [
            pd.DataFrame(X_resampled, columns=X.columns),
            pd.Series(y_resampled, name="classification")
        ],
        axis=1
    )
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced_df

# display class distribution
def show_distribution(df, name):
    print(f"\n{name} Distribution:")
    print(df["classification"].value_counts(normalize=True))

# save hospital datasets
def save_hospital_data(datasets, output_base):
    for name, (train_df, val_df, test_df) in datasets.items():
        hospital_path = os.path.join(output_base, name)
        train_df.to_csv(f"{hospital_path}/train.csv", index=False)
        val_df.to_csv(f"{hospital_path}/val.csv", index=False)
        test_df.to_csv(f"{hospital_path}/test.csv", index=False)

# create hospitals using UCI + synthetic ratio
def create_hospitals(uci, synthetic):
    uci = uci.sample(frac=1, random_state=42).reset_index(drop=True)
    synthetic = synthetic.sample(frac=1, random_state=42).reset_index(drop=True)

    uci_A = uci.iloc[:130]
    uci_B = uci.iloc[130:260]
    uci_C = uci.iloc[260:400]

    syn_A = synthetic.iloc[:100]
    syn_B = synthetic.iloc[100:200]
    syn_C = synthetic.iloc[200:300]

    hospital_A = pd.concat([uci_A, syn_A]).sample(frac=1, random_state=42).reset_index(drop=True)
    hospital_B = pd.concat([uci_B, syn_B]).sample(frac=1, random_state=42).reset_index(drop=True)
    hospital_C = pd.concat([uci_C, syn_C]).sample(frac=1, random_state=42).reset_index(drop=True)

    return hospital_A, hospital_B, hospital_C

# main hospital simulation pipeline
def run_hospital_simulation():
    print("HOSPITAL SIMULATION..........")
    uci_clean, synthetic_clean = load_datasets()
    hospital_A, hospital_B, hospital_C = create_hospitals(uci_clean, synthetic_clean)

    print("\nPerforming train/val/test splits...")
    A_train, A_val, A_test = train_val_test_split(hospital_A)
    B_train, B_val, B_test = train_val_test_split(hospital_B)
    C_train, C_val, C_test = train_val_test_split(hospital_C)

    A_train = handle_missing_values(A_train)
    B_train = handle_missing_values(B_train)
    C_train = handle_missing_values(C_train)

    A_train_before = A_train.copy()
    B_train_before = B_train.copy()
    C_train_before = C_train.copy()

    print("\nApplying SMOTE to balance training datasets...")
    A_train = apply_smote(A_train)
    B_train = apply_smote(B_train)
    C_train = apply_smote(C_train)

    output_base = "../data/processed"
    hospital_names = ["hospital_A", "hospital_B", "hospital_C"]
    create_directories(output_base, hospital_names)
    datasets = {
        "hospital_A": (A_train, A_val, A_test),
        "hospital_B": (B_train, B_val, B_test),
        "hospital_C": (C_train, C_val, C_test)
    }
    save_hospital_data(datasets, output_base)

    print("\nHospital datasets saved successfully!")
    print("\n--- Dataset Summary ---")

    print(f"Hospital A - Train: {A_train.shape[0]}, Val: {A_val.shape[0]}, Test: {A_test.shape[0]}")
    print(f"Hospital B - Train: {B_train.shape[0]}, Val: {B_val.shape[0]}, Test: {B_test.shape[0]}")
    print(f"Hospital C - Train: {C_train.shape[0]}, Val: {C_val.shape[0]}, Test: {C_test.shape[0]}")

    show_distribution(A_train, "Hospital-A Train")
    show_distribution(B_train, "Hospital-B Train")
    show_distribution(C_train, "Hospital-C Train")

    print("\nGenerating SMOTE comparison plot...")
    plot_smote_comparison(
        train_before=[A_train_before, B_train_before, C_train_before],
        train_after=[A_train, B_train, C_train]
    )

if __name__ == "__main__":
    run_hospital_simulation()