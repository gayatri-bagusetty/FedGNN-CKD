import numpy as np
import pandas as pd
import torch
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

# -------------------------------------------------
# CORE GRAPH BUILDER (shared)
# -------------------------------------------------
def build_graph(X, y, k=5):
    """Construct PyTorch Geometric graph from features X and labels y."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    n_samples = X.shape[0]
    k = min(k, n_samples - 1)

    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(X)
    _, indices = knn.kneighbors(X)

    edge_index = []
    for i in range(n_samples):
        for j in indices[i]:
            if i != j:
                edge_index.append([i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(
        x=torch.tensor(X, dtype=torch.float),
        y=torch.tensor(y, dtype=torch.long),
        edge_index=edge_index
    )

# -------------------------------------------------
# DASHBOARD / UPLOADED DATA GRAPH
# -------------------------------------------------
def build_graph_from_uploaded(df: pd.DataFrame, k=5):
    """
    Build a graph from uploaded/admin DataFrame.
    Returns PyTorch Geometric Data object.
    """
    print("[INFO] Building graph from uploaded/admin DataFrame...")
    X = df.drop("classification", axis=1)
    y = df["classification"]

    graph = build_graph(X, y, k)
    print(f"[INFO] Graph created: Nodes={graph.num_nodes}, Edges={graph.num_edges}")

    return graph

# -------------------------------------------------
# HOSPITAL SIMULATION (Default datasets)
# -------------------------------------------------
def simulate_hospitals():
    """Split default datasets into hospital-wise train/val CSVs."""
    print("[INFO] Simulating hospitals with default datasets...")

    uci_clean = pd.read_csv("../data/processed/uci_clean.csv")
    kaggle_clean = pd.read_csv("../data/processed/kaggle_clean.csv")

    # Assign Hospital-A (UCI CKD)
    hospital_A = uci_clean.copy()

    # Split Kaggle for Hospital-B & C
    ckd_data = kaggle_clean[kaggle_clean['classification'] == 1]
    non_ckd_data = kaggle_clean[kaggle_clean['classification'] == 0]

    hospital_B = pd.concat([
        ckd_data.sample(frac=0.7, random_state=42),
        non_ckd_data.sample(frac=0.3, random_state=42)
    ]).sample(frac=1, random_state=42)

    hospital_C = kaggle_clean.drop(hospital_B.index)

    def local_split(df):
        return train_test_split(df, test_size=0.2, stratify=df['classification'], random_state=42)

    A_train, A_val = local_split(hospital_A)
    B_train, B_val = local_split(hospital_B)
    C_train, C_val = local_split(hospital_C)

    output_base = "../data/processed"
    hospitals = ["hospital_A", "hospital_B", "hospital_C"]
    for h in hospitals:
        os.makedirs(os.path.join(output_base, h), exist_ok=True)

    datasets = {
        "hospital_A": (A_train, A_val),
        "hospital_B": (B_train, B_val),
        "hospital_C": (C_train, C_val)
    }

    for name, (train_df, val_df) in datasets.items():
        train_df.to_csv(f"{output_base}/{name}/train.csv", index=False)
        val_df.to_csv(f"{output_base}/{name}/val.csv", index=False)

    print("[INFO] Hospital CSV files saved successfully.")
    return datasets  # Optional: can return train/val dataframes if needed

# -------------------------------------------------
# DEFAULT HOSPITAL GRAPH CONSTRUCTION
# -------------------------------------------------
def build_graphs_from_default(k=5):
    """
    Build graphs for default hospitals.
    Runs hospital simulation first to create train CSVs.
    Saves graphs to ../data/graph/ and returns dict of graphs.
    """
    print("[INFO] Building graphs from default hospital datasets...")

    # 1️⃣ Simulate hospitals
    simulate_hospitals()

    input_base = "../data/processed"
    output_path = "../data/graph"
    hospitals = ["hospital_A", "hospital_B", "hospital_C"]

    os.makedirs(output_path, exist_ok=True)

    hospital_graphs = {}

    for hospital in hospitals:
        file_path = f"{input_base}/{hospital}/train.csv"
        if not os.path.exists(file_path):
            print(f"[WARNING] {file_path} not found — skipping")
            continue

        df = pd.read_csv(file_path)
        X = df.drop("classification", axis=1)
        y = df["classification"]

        graph = build_graph(X, y, k)

        save_file = f"{output_path}/graph_{hospital.split('_')[1]}.pt"
        torch.save(graph, save_file)

        print(
            f"[INFO] Saved graph_{hospital.split('_')[1]}.pt "
            f"(Nodes={graph.num_nodes}, Edges={graph.num_edges})"
        )

        hospital_graphs[hospital.split('_')[1]] = graph

    return hospital_graphs

# -------------------------------------------------
# TEST
# -------------------------------------------------
if __name__ == "__main__":
    # Example: uploaded/admin data
    # df = pd.read_csv("path_to_uploaded_file.csv")
    # graph = build_graph_from_uploaded(df)

    # Default hospital graphs
    graphs = build_graphs_from_default()
    print("Hospital graphs ready:", graphs.keys())