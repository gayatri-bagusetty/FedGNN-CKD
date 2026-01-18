import numpy as np
import pandas as pd
import torch
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from torch_geometric.data import Data

def build_graph(X, y, k=5):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    # Impute to handle any edge cases in synthetic data
    imputer = SimpleImputer(strategy="constant", fill_value=0)
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

    graph = Data(
        x=torch.tensor(X, dtype=torch.float),
        y=torch.tensor(y, dtype=torch.long),
        edge_index=edge_index
    )
    return graph

def main():
    # Paths according to hospital_simulation.py output
    input_base_path = "../data/processed"
    output_path = "../data/graph"
    os.makedirs(output_path, exist_ok=True)

    # UPDATED: Mapping to specific simulation files
    hospitals = {
        "A": "hospital_A/train.csv",
        "B": "hospital_B/train.csv",
        "C": "hospital_C/train.csv"
    }

    for h_id, relative_path in hospitals.items():
        file_path = os.path.join(input_base_path, relative_path)
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. Skipping...")
            continue

        print(f"--- Building Graph for Hospital {h_id} ---")
        df = pd.read_csv(file_path)
        
        X = df.drop("classification", axis=1)
        y = df["classification"]
        
        graph = build_graph(X, y, k=5)
        
        save_name = f"graph_{h_id}.pt"
        save_path = os.path.join(output_path, save_name)
        torch.save(graph, save_path)
        
        print(f"Successfully saved graph_{h_id}.pt | Nodes: {graph.num_nodes} | Features: {graph.num_node_features}")

if __name__ == "__main__":
    main()