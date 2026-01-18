import os
import pandas as pd
from sklearn.model_selection import train_test_split

def create_directories(base_path, hospitals):
    """Ensures the directory structure exists for saving CSVs."""
    for hospital in hospitals:
        path = os.path.join(base_path, hospital)
        os.makedirs(path, exist_ok=True)

def local_split(df):
    """Splits a dataframe into 80% train and 20% validation."""
    return train_test_split(
        df, 
        test_size=0.2, 
        stratify=df['classification'], 
        random_state=42
    )

def show_dist(df, name):
    """Prints the distribution of the target class."""
    print(f"\n{name} Distribution:")
    print(df['classification'].value_counts(normalize=True))

def main():
    # 1. Load Data from processed folder
    print("Loading processed hospital datasets...")
    try:
        uci_clean = pd.read_csv("../data/processed/uci_clean.csv")
        kaggle_clean = pd.read_csv("../data/processed/kaggle_clean.csv")
        synthetic_clean = pd.read_csv("../data/processed/synthetic_clean.csv")
    except FileNotFoundError as e:
        print(f"Error: Could not find processed files. Run preprocessing.py first. {e}")
        return

    # 2. Direct Assignment (Matching your new requirement)
    # Hospital A - UCI Data
    hospital_A = uci_clean.copy()

    # Hospital B - Kaggle Data
    hospital_B = kaggle_clean.copy()

    # Hospital C - Synthetic Data
    hospital_C = synthetic_clean.copy()

    # 3. Perform Train/Validation Splits for each hospital
    print("Performing local train/val splits...")
    A_train, A_val = local_split(hospital_A)
    B_train, B_val = local_split(hospital_B)
    C_train, C_val = local_split(hospital_C)

    # 4. Create Directories and Save Files
    output_base = "../data/processed"
    hospital_names = ['hospital_A', 'hospital_B', 'hospital_C']
    create_directories(output_base, hospital_names)

    datasets = {
        "hospital_A": (A_train, A_val),
        "hospital_B": (B_train, B_val),
        "hospital_C": (C_train, C_val)
    }

    for name, (train_df, val_df) in datasets.items():
        train_df.to_csv(f"{output_base}/{name}/train.csv", index=False)
        val_df.to_csv(f"{output_base}/{name}/val.csv", index=False)
    
    print("\nHospital-specific datasets saved successfully in ../data/processed/")

    # 5. Verify and Show Distribution
    print("\n--- Summary ---")
    print(f"Hospital A (UCI)       - Train: {A_train.shape[0]}, Val: {A_val.shape[0]}")
    print(f"Hospital B (Kaggle)    - Train: {B_train.shape[0]}, Val: {B_val.shape[0]}")
    print(f"Hospital C (Synthetic) - Train: {C_train.shape[0]}, Val: {C_val.shape[0]}")

    show_dist(A_train, "Hospital-A (UCI)")
    show_dist(B_train, "Hospital-B (Kaggle)")
    show_dist(C_train, "Hospital-C (Synthetic)")

if __name__ == "__main__":
    main()