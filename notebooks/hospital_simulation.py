import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from source.plot_hospital_distribution import plot_train_distribution


# LOAD CLEAN DATASETS
def load_datasets():

    print("\nLoading processed hospital datasets...")

    try:
        uci_clean = pd.read_csv("../data/processed/uci_clean.csv")
        kaggle_clean = pd.read_csv("../data/processed/kaggle_clean.csv")
        synthetic_clean = pd.read_csv("../data/processed/synthetic_clean.csv")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "Processed files not found. Run preprocessing.py first."
        ) from e

    return uci_clean, kaggle_clean, synthetic_clean


# CREATE HOSPITAL DIRECTORIES
def create_directories(base_path, hospitals):

    for hospital in hospitals:
        path = os.path.join(base_path, hospital)
        os.makedirs(path, exist_ok=True)


# TRAIN / VAL / TEST SPLIT
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


# HANDLE MISSING VALUES
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


# APPLY SMOTE ON TRAIN DATA ONLY
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

    # Shuffle dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced_df


# DISPLAY CLASS DISTRIBUTION
def show_distribution(df, name):

    print(f"\n{name} Distribution:")
    print(df["classification"].value_counts(normalize=True))


# SAVE DATASETS
def save_hospital_data(datasets, output_base):

    for name, (train_df, val_df, test_df) in datasets.items():

        hospital_path = os.path.join(output_base, name)

        train_df.to_csv(f"{hospital_path}/train.csv", index=False)
        val_df.to_csv(f"{hospital_path}/val.csv", index=False)
        test_df.to_csv(f"{hospital_path}/test.csv", index=False)


# PLOT TRAIN DISTRIBUTION
def plot_distributions(A_train, B_train, C_train):

    dist_dict = {
        "Hospital A": A_train["classification"].value_counts(normalize=True).to_dict(),
        "Hospital B": B_train["classification"].value_counts(normalize=True).to_dict(),
        "Hospital C": C_train["classification"].value_counts(normalize=True).to_dict(),
    }

    plot_train_distribution(dist_dict)


# MAIN PIPELINE
def run_hospital_simulation():

    print("HOSPITAL SIMULATION..........")

    # Load datasets
    uci_clean, kaggle_clean, synthetic_clean = load_datasets()

    # Assign hospitals
    hospital_A = uci_clean.copy()
    hospital_B = kaggle_clean.copy()
    hospital_C = synthetic_clean.copy()

    print("\nPerforming train/val/test splits...")

    A_train, A_val, A_test = train_val_test_split(hospital_A)
    B_train, B_val, B_test = train_val_test_split(hospital_B)
    C_train, C_val, C_test = train_val_test_split(hospital_C)

    # Handle missing values BEFORE SMOTE
    A_train = handle_missing_values(A_train)
    B_train = handle_missing_values(B_train)
    C_train = handle_missing_values(C_train)

    # Apply SMOTE
    print("\nApplying SMOTE to balance training datasets...")

    A_train = apply_smote(A_train)
    B_train = apply_smote(B_train)
    C_train = apply_smote(C_train)

    # Output directories
    output_base = "../data/processed"
    hospital_names = ["hospital_A", "hospital_B", "hospital_C"]

    create_directories(output_base, hospital_names)

    datasets = {
        "hospital_A": (A_train, A_val, A_test),
        "hospital_B": (B_train, B_val, B_test),
        "hospital_C": (C_train, C_val, C_test)
    }

    # Save datasets
    save_hospital_data(datasets, output_base)

    print("\nHospital datasets saved successfully!")

    print("\n--- Dataset Summary ---")

    print(f"Hospital A - Train: {A_train.shape[0]}, Val: {A_val.shape[0]}, Test: {A_test.shape[0]}")
    print(f"Hospital B - Train: {B_train.shape[0]}, Val: {B_val.shape[0]}, Test: {B_test.shape[0]}")
    print(f"Hospital C - Train: {C_train.shape[0]}, Val: {C_val.shape[0]}, Test: {C_test.shape[0]}")

    # Show distributions
    show_distribution(A_train, "Hospital-A Train")
    show_distribution(B_train, "Hospital-B Train")
    show_distribution(C_train, "Hospital-C Train")

    # Plot distribution
    plot_distributions(A_train, B_train, C_train)

    print("Plotting is done.........." )


# RUN
if __name__ == "__main__":
    run_hospital_simulation()