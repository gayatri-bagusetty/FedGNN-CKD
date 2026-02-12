import os
import pandas as pd
from sklearn.model_selection import train_test_split


def create_directories(base_path, hospitals):
    """Ensures the directory structure exists for saving CSVs."""
    for hospital in hospitals:
        path = os.path.join(base_path, hospital)
        os.makedirs(path, exist_ok=True)


def train_val_test_split(df):
    """Splits dataframe into 70% train, 15% val, 15% test."""
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df['classification'],
        random_state=42
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df['classification'],
        random_state=42
    )

    return train_df, val_df, test_df


def show_dist(df, name):
    print(f"\n{name} Distribution:")
    print(df['classification'].value_counts(normalize=True))


def main():

    print("Loading processed hospital datasets...")
    try:
        uci_clean = pd.read_csv("../data/processed/uci_clean.csv")
        kaggle_clean = pd.read_csv("../data/processed/kaggle_clean.csv")
        synthetic_clean = pd.read_csv("../data/processed/synthetic_clean.csv")
    except FileNotFoundError as e:
        print(f"Error: Could not find processed files. Run preprocessing.py first. {e}")
        return

    # Hospital assignment
    hospital_A = uci_clean.copy()
    hospital_B = kaggle_clean.copy()
    hospital_C = synthetic_clean.copy()

    print("Performing train/val/test splits...")
    A_train, A_val, A_test = train_val_test_split(hospital_A)
    B_train, B_val, B_test = train_val_test_split(hospital_B)
    C_train, C_val, C_test = train_val_test_split(hospital_C)

    output_base = "../data/processed"
    hospital_names = ['hospital_A', 'hospital_B', 'hospital_C']
    create_directories(output_base, hospital_names)

    datasets = {
        "hospital_A": (A_train, A_val, A_test),
        "hospital_B": (B_train, B_val, B_test),
        "hospital_C": (C_train, C_val, C_test)
    }

    for name, (train_df, val_df, test_df) in datasets.items():
        train_df.to_csv(f"{output_base}/{name}/train.csv", index=False)
        val_df.to_csv(f"{output_base}/{name}/val.csv", index=False)
        test_df.to_csv(f"{output_base}/{name}/test.csv", index=False)

    print("\nHospital-specific datasets (train/val/test) saved successfully!")

    print("\n--- Summary ---")
    print(f"Hospital A - Train: {A_train.shape[0]}, Val: {A_val.shape[0]}, Test: {A_test.shape[0]}")
    print(f"Hospital B - Train: {B_train.shape[0]}, Val: {B_val.shape[0]}, Test: {B_test.shape[0]}")
    print(f"Hospital C - Train: {C_train.shape[0]}, Val: {C_val.shape[0]}, Test: {C_test.shape[0]}")

    show_dist(A_train, "Hospital-A Train")
    show_dist(B_train, "Hospital-B Train")
    show_dist(C_train, "Hospital-C Train")


if __name__ == "__main__":
    main()