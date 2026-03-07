import os
import torch
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Load Dataset
def load_hospital_data(file_path):
    df = pd.read_csv(file_path)
    if "classification" not in df.columns:
        raise ValueError("Target column 'classification' not found")

    X = df.drop(columns=["classification"])
    y = df["classification"]
    return X, y

# Normalize Features
def normalize_features(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

# Feature Similarity
def compute_feature_similarity(X):
    sim = cosine_similarity(X)
    sim = (sim + 1) / 2  # normalize to [0,1]
    return sim

# Clinical Similarity
def compute_clinical_similarity(X, clinical_indices):
    clinical_features = X[:, clinical_indices]
    sim = cosine_similarity(clinical_features)
    sim = (sim + 1) / 2
    return sim

# Hybrid Similarity
def compute_hybrid_similarity(feature_sim, clinical_sim, alpha=0.6):
    hybrid = alpha * feature_sim + (1 - alpha) * clinical_sim

    # Safety clipping
    hybrid = np.clip(hybrid, 0, 1)
    return hybrid

# Build KNN Graph
def build_hybrid_graph(sim_matrix, k=8):
    n_nodes = sim_matrix.shape[0]

    # Convert similarity to distance
    distance_matrix = 1 - sim_matrix
    nbrs = NearestNeighbors(
        n_neighbors=k + 1,
        metric="precomputed"
    )
    nbrs.fit(distance_matrix)
    distances, indices = nbrs.kneighbors(distance_matrix)
    edge_index = []
    for i in range(n_nodes):
        for j in indices[i][1:]:  # skip self
            edge_index.append([i, j])
            edge_index.append([j, i])  # make graph undirected
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    # Remove duplicates
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index

# Create PyG Graph
def create_pyg_graph(X, y, edge_index):
    x = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y.values, dtype=torch.long)
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y
    )
    return data

# single graph connstruction
def build_patient_similarity_graph(
        new_patient,
        train_features,
        train_edge_index,
        k=5):
    """
    Build graph by connecting new patient to K nearest training patients.
    """
    # Convert to numpy
    new_patient_np = new_patient.cpu().numpy()
    # Fit KNN on training patients
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(train_features)
    distances, indices = knn.kneighbors(new_patient_np)
    # Convert indices to list
    neighbor_ids = indices[0]
    # New node index
    new_node_index = train_features.shape[0]
    new_edges = []
    for neighbor in neighbor_ids:
        # patient -> neighbor
        new_edges.append([new_node_index, neighbor])
        # neighbor -> patient (undirected)
        new_edges.append([neighbor, new_node_index])
    new_edges = torch.tensor(new_edges).t().long()

    # Combine edges
    edge_index = torch.cat([train_edge_index, new_edges], dim=1)

    # Combine node features
    x = torch.cat(
        [
            torch.tensor(train_features, dtype=torch.float32),
            new_patient
        ],
        dim=0
    )
    return x, edge_index, new_node_index

# Graph Construction Pipeline
def construct_graph(file_path, k=8):
    print(f"\nProcessing {file_path}")
    X, y = load_hospital_data(file_path)
    X_scaled = normalize_features(X)

    # Feature Similarity
    feature_sim = compute_feature_similarity(X_scaled)

    # Clinical Features (important CKD attributes)
    clinical_features = ["age", "bp", "sg", "al", "su"]

    clinical_indices = [
        X.columns.get_loc(col)
        for col in clinical_features
        if col in X.columns
    ]

    clinical_sim = compute_clinical_similarity(
        X_scaled,
        clinical_indices
    )
    # Hybrid Similarity
    hybrid_sim = compute_hybrid_similarity(
        feature_sim,
        clinical_sim
    )
    # Graph Construction
    edge_index = build_hybrid_graph(
        hybrid_sim,
        k
    )
    graph = create_pyg_graph(
        X_scaled,
        y,
        edge_index
    )
    print("Graph Created")
    print(graph)
    return graph

# Save Graph
def save_graph(graph, save_path):
    torch.save(graph, save_path)
    print(f"Graph saved at {save_path}")

# Build Graphs for All Hospitals
def build_all_hospital_graphs():
    input_base = "../data/processed"
    output_base = "../data/graph"
    os.makedirs(output_base, exist_ok=True)

    hospitals = [
        "hospital_A",
        "hospital_B",
        "hospital_C"
    ]

    splits = [
        "train",
        "val",
        "test"
    ]
    
    for hospital in hospitals:
        for split in splits:
            file_path = os.path.join(
                input_base,
                hospital,
                f"{split}.csv"
            )
            if not os.path.exists(file_path):
                print(f"Skipping {file_path}")
                continue
            graph = construct_graph(file_path)
            save_name = f"{hospital}_{split}.pt"
            save_path = os.path.join(
                output_base,
                save_name
            )
            save_graph(graph, save_path)

# Main
if __name__ == "__main__":
    set_seed(42)
    build_all_hospital_graphs()