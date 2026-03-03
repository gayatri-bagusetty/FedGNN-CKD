import os
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from torch_geometric.utils import add_self_loops

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def build_single_node_graph(X, y):
    return Data(
        x=torch.tensor(X, dtype=torch.float),
        y=torch.tensor(y, dtype=torch.long),
        edge_index=torch.tensor([[0], [0]], dtype=torch.long)
    )


def build_graph(X, y, k=20):

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    n_samples = X.shape[0]

    if n_samples <= 1:
        return build_single_node_graph(X, y)

    k = min(k + 1, n_samples)

    knn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="brute")
    knn.fit(X)
    _, indices = knn.kneighbors(X)

    edge_index = []

    for i in range(n_samples):
        for j in indices[i][1:]:   # skip self-loop
            edge_index.append([i, j])
            edge_index.append([j, i])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_index, _ = add_self_loops(edge_index, num_nodes=n_samples)

    return Data(
        x=torch.tensor(X, dtype=torch.float),
        y=torch.tensor(y, dtype=torch.long),
        edge_index=edge_index
    )


def visualize_graph(data, save_path, max_nodes=300, title="Graph"):

    G = to_networkx(data, to_undirected=True)

    if G.number_of_nodes() > max_nodes:
        nodes = list(G.nodes)[:max_nodes]
        G = G.subgraph(nodes)

    labels = data.y[:G.number_of_nodes()].cpu().numpy()
    unique_labels = np.unique(labels)
    cmap = cm.get_cmap("tab10", 5)
    colors = [cmap(int(l)) for l in labels]

    degrees = dict(G.degree())
    sizes = [degrees[n] * 20 for n in G.nodes()]

    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G, seed=42)

    nx.draw(
        G,
        pos,
        node_size=sizes,
        node_color=colors,
        edge_color="gray",
        alpha=0.8,
        with_labels=False
    )
    
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
           label=f'Class {i}',
           markerfacecolor=cmap(i),
           markersize=8)
        for i in range(5)
    ]

    # plt.legend(handles=legend_elements, loc="best")
    plt.title(title)
    plt.savefig(save_path.replace(".pt", ".png"), dpi=300, bbox_inches="tight")
    plt.close()


def process_csv(file_path, save_path, title, max_nodes=100):

    df = pd.read_csv(file_path)

    if "target" not in df.columns:
        raise ValueError("Column 'target' not found in hospital CSV.")

    X = df.drop("target", axis=1).values
    y = df["target"].values

    graph = build_graph(X, y, k=20)

    torch.save(graph, save_path)

    visualize_graph(graph, save_path, max_nodes=max_nodes, title=title)

    print(f"Stored graph and figure in {os.path.dirname(save_path)}")


def main():

    print("GRAPH CONSTRUCTION OF EACH HOSPITAL.............")

    input_base_path = "../data/processed"
    output_base_path = "../data/graph"
    test_output_path = os.path.join(output_base_path, "test")

    os.makedirs(output_base_path, exist_ok=True)
    os.makedirs(test_output_path, exist_ok=True)

    hospitals = ["hospital_A", "hospital_B", "hospital_C"]
    splits = ["train", "val", "test"]

    for hospital in hospitals:

        hospital_path = os.path.join(input_base_path, hospital)

        for split in splits:

            file_path = os.path.join(hospital_path, f"{split}.csv")

            if not os.path.exists(file_path):
                continue

            print(f"{hospital} {split}.csv graph building...")

            if split == "test":
                save_path = os.path.join(test_output_path, f"{hospital}_test.pt")
            else:
                save_path = os.path.join(output_base_path, f"{hospital}_{split}.pt")

            title = f"{hospital.upper()} - {split.upper()} Graph"

            process_csv(file_path, save_path, title, max_nodes=100)

    print("All hospital graphs created successfully.")


if __name__ == "__main__":
    main()