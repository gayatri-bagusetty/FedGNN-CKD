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
    # 1. Load Data
    print("Loading raw processed data...")
    uci_clean = pd.read_csv("../data/processed/uci_clean.csv")
    kaggle_clean = pd.read_csv("../data/processed/kaggle_clean.csv")

    # 2. Assign Hospital-A (UCI CKD)
    hospital_A = uci_clean.copy()

    # 3. Split Kaggle Dataset → Hospital-B & C (Simulating Non-IID skew)
    # Separate Kaggle data by class
    ckd_data = kaggle_clean[kaggle_clean['classification'] == 1]
    non_ckd_data = kaggle_clean[kaggle_clean['classification'] == 0]

    # Hospital-B: CKD-heavy hospital (70% of all CKD cases, 30% of all non-CKD)
    hospital_B = pd.concat([
        ckd_data.sample(frac=0.7, random_state=42),
        non_ckd_data.sample(frac=0.3, random_state=42)
    ]).sample(frac=1, random_state=42)  # shuffle

    # Hospital-C: remaining data (Non-CKD heavy)
    hospital_C = kaggle_clean.drop(hospital_B.index)

    # 4. Perform Train/Validation Splits
    print("Splitting data into train/val sets...")
    A_train, A_val = local_split(hospital_A)
    B_train, B_val = local_split(hospital_B)
    C_train, C_val = local_split(hospital_C)

    # 5. Create Directories and Save Files
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
    
    print("Files saved successfully.")

    # 6. Verify and Show Distribution
    print("\n--- Summary ---")
    print(f"Hospital A (Train/Val): {A_train.shape[0]} / {A_val.shape[0]}")
    print(f"Hospital B (Train/Val): {B_train.shape[0]} / {B_val.shape[0]}")
    print(f"Hospital C (Train/Val): {C_train.shape[0]} / {C_val.shape[0]}")

    show_dist(A_train, "Hospital-A (UCI)")
    show_dist(B_train, "Hospital-B (Kaggle)")
    show_dist(C_train, "Hospital-C (Kaggle)")

if __name__ == "__main__":
    main()