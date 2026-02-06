import numpy as np
import pandas as pd
import torch
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

def build_single_node_graph(x_tensor):
    edge_index = torch.tensor([[0],[0]], dtype=torch.long)
    return x_tensor, edge_index

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

def visualize_graph(graph, title, save_path, max_nodes=100, show=False):
    """
    Visualize PyG graph using NetworkX and save image
    """
    g_nx = to_networkx(graph, to_undirected=True)

    if g_nx.number_of_nodes() > max_nodes:
        g_nx = g_nx.subgraph(list(g_nx.nodes)[:max_nodes])

    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(g_nx, seed=42)
    nx.draw(
        g_nx,
        pos,
        node_size=50,
        node_color="skyblue",
        edge_color="gray",
        with_labels=False
    )
    plt.title(title)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    # if show:
    #     plt.show()
    # else:
    #     plt.close()


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
        img_path = os.path.join(output_path, f"graph_{h_id}.png")
        visualize_graph(
            graph,
            title=f"Hospital {h_id} Graph",
            save_path=img_path,
            show=True
        )
        print(f"Graph images are saved")


if __name__ == "__main__":
    main()