import numpy as np
import pandas as pd
import torch
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from torch_geometric.data import Data

def build_graph(X, y, k=5):
    """
    Constructs a PyTorch Geometric graph where nodes are patients 
    and edges represent clinical similarity via KNN.
    """
    # Convert to NumPy arrays
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    # Handle missing values (Ensures no NaN values enter the graph)
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    # KNN-based graph construction
    n_samples = X.shape[0]
    k = min(k, n_samples - 1)

    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(X)
    _, indices = knn.kneighbors(X)

    # Build edge index (Source to Target node mapping)
    edge_index = []
    for i in range(n_samples):
        for j in indices[i]:
            if i != j:
                edge_index.append([i, j])

    # Transpose to meet PyG format [2, num_edges]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Create PyTorch Geometric Data object
    graph = Data(
        x=torch.tensor(X, dtype=torch.float),
        y=torch.tensor(y, dtype=torch.long),
        edge_index=edge_index
    )

    return graph

def split_features_labels(df):
    """Splits dataframe into features (X) and target label (y)."""
    X = df.drop("classification", axis=1)
    y = df["classification"]
    return X, y

def main():
    # 1. Define Paths
    input_base_path = "../data/processed"
    output_path = "../data/graph"
    hospitals = ["hospital_A", "hospital_B", "hospital_C"]
    
    os.makedirs(output_path, exist_ok=True)

    # 2. Process each hospital
    for hospital in hospitals:
        file_path = f"{input_base_path}/{hospital}/train.csv"
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. Skipping...")
            continue

        print(f"--- Building Graph for {hospital} ---")
        df = pd.read_csv(file_path)
        X, y = split_features_labels(df)
        
        # Build the graph
        graph = build_graph(X, y, k=5)
        
        # Save the .pt file
        save_file = f"{output_path}/graph_{hospital.split('_')[1]}.pt"
        torch.save(graph, save_file)
        
        print(f"Successfully saved {hospital} graph:")
        print(f"  Nodes: {graph.num_nodes}")
        print(f"  Edges: {graph.num_edges}")

if __name__ == "__main__":
    main()