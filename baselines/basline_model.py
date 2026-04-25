import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# LOAD HOSPITAL DATA
def load_hospital_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    target_col = "classification"

    X_train = train_df.drop(target_col, axis=1).values
    y_train = train_df[target_col].values

    X_test = test_df.drop(target_col, axis=1).values
    y_test = test_df[target_col].values

    # SCALE FEATURES
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# MODEL EVALUATION
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs)
    }

# RUN BASELINES
def run_baselines(train_path, test_path):
    X_train, X_test, y_train, y_test = load_hospital_data(train_path, test_path)
    results = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=3000)
    lr.fit(X_train, y_train)
    results["Logistic Regression"] = evaluate_model(lr, X_test, y_test)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    results["Random Forest"] = evaluate_model(rf, X_test, y_test)

    # MLP
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    mlp.fit(X_train, y_train)
    results["MLP"] = evaluate_model(mlp, X_test, y_test)

    results_df = pd.DataFrame(results).T
    return results_df

# RUN FOR ALL HOSPITALS
if __name__ == "__main__":
    hospitals = ["hospital_A", "hospital_B", "hospital_C"]
    all_results = {}
    for hospital in hospitals:
        print(f"\nRunning Baselines for {hospital}")
        train_path = f"../data/processed/{hospital}/train.csv"
        test_path = f"../data/processed/{hospital}/test.csv"
        results = run_baselines(train_path, test_path)
        print(results)
        all_results[hospital] = results
    # SAVE FINAL RESULTS
    os.makedirs("plots", exist_ok=True)
    for hospital, df in all_results.items():
        df.to_csv(f"plots/{hospital}_baseline_results.csv")
    print("\nBaseline results saved in plots/ folder")