import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def create_directories(base_path, hospitals):
    for hospital in hospitals:
        path = os.path.join(base_path, hospital)
        os.makedirs(path, exist_ok=True)


def split_hospitals(X, y, num_hospitals=3):
    indices = np.random.permutation(len(X))
    splits = np.array_split(indices, num_hospitals)

    hospital_data = []

    for idx in splits:
        hospital_data.append((X[idx], y[idx]))

    return hospital_data


def save_splits(X, y, hospital_name, base_path):

    # 70 / 15 / 15 split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        stratify=y,
        random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=42
    )

    df_train = pd.DataFrame(X_train)
    df_train["target"] = y_train

    df_val = pd.DataFrame(X_val)
    df_val["target"] = y_val

    df_test = pd.DataFrame(X_test)
    df_test["target"] = y_test

    df_train.to_csv(f"{base_path}/{hospital_name}/train.csv", index=False)
    df_val.to_csv(f"{base_path}/{hospital_name}/val.csv", index=False)
    df_test.to_csv(f"{base_path}/{hospital_name}/test.csv", index=False)

    print(f"{hospital_name} - Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")


def main():
    print("HOSPITAL SIMULATION.................")
    print("Loading processed dataset...")

    X = np.load("../data/processed/X.npy")
    y = np.load("../data/processed/y.npy")

    print("Dataset shape:", X.shape)

    hospital_data = split_hospitals(X, y, num_hospitals=3)

    output_base = "../data/processed"
    hospital_names = ['hospital_A', 'hospital_B', 'hospital_C']
    create_directories(output_base, hospital_names)

    for i, (X_h, y_h) in enumerate(hospital_data):
        save_splits(X_h, y_h, hospital_names[i], output_base)

    print("\nHospital datasets created successfully.")


if __name__ == "__main__":
    main()