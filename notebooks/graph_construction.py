import numpy as np
import pandas as pd
import torch
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from torch_geometric.data import Data

def build_single_node_graph(x_tensor):
    edge_index = torch.tensor([[0],[0]], dtype=torch.long)
    return x_tensor, edge_index

def build_graph(X, y, k=5):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

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


def visualize_graph(graph, title, save_path, max_nodes=100):
    import matplotlib.pyplot as plt
    import networkx as nx
    from torch_geometric.utils import to_networkx

    g_nx = to_networkx(graph, to_undirected=True)

    if g_nx.number_of_nodes() > max_nodes:
        g_nx = g_nx.subgraph(list(g_nx.nodes)[:max_nodes])

    if hasattr(graph, "y"):
        labels = graph.y[:g_nx.number_of_nodes()].cpu().numpy()
        colors = ["red" if l == 1 else "green" for l in labels]
    else:
        colors = "skyblue"

    degrees = dict(g_nx.degree())
    sizes = [degrees[n] * 30 for n in g_nx.nodes()]

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(g_nx, seed=42)

    nx.draw(
        g_nx,
        pos,
        node_size=sizes,
        node_color=colors,
        edge_color="gray",
        alpha=0.8,
        with_labels=False
    )

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='CKD',
               markerfacecolor='red', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Non-CKD',
               markerfacecolor='green', markersize=8)
    ]
    plt.legend(handles=legend_elements, loc="best")

    plt.title(title, fontsize=12)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    input_base_path = "../data/processed"
    output_base_path = "../data/graph"
    test_output_path = os.path.join(output_base_path, "test")

    os.makedirs(output_base_path, exist_ok=True)
    os.makedirs(test_output_path, exist_ok=True)

    hospitals = ["hospital_A", "hospital_B", "hospital_C"]

    for hospital in hospitals:

        hospital_path = os.path.join(input_base_path, hospital)

        if not os.path.exists(hospital_path):
            print(f"{hospital} folder not found. Skipping...")
            continue

        for split in ["train", "val", "test"]:

            file_path = os.path.join(hospital_path, f"{split}.csv")

            if not os.path.exists(file_path):
                continue

            print(f"{hospital} {split}.csv graph building...")

            df = pd.read_csv(file_path)

            X = df.drop("classification", axis=1)
            y = df["classification"]

            graph = build_graph(X, y, k=5)

            # ---------- SAVE PATH LOGIC ----------
            if split == "test":
                save_name = f"{hospital}_test.pt"
                save_path = os.path.join(test_output_path, save_name)
                img_path = os.path.join(test_output_path, f"{hospital}_test.png")
            else:
                save_name = f"{hospital}_{split}.pt"
                save_path = os.path.join(output_base_path, save_name)
                img_path = os.path.join(output_base_path, f"{hospital}_{split}.png")
            # -------------------------------------

            torch.save(graph, save_path)

            visualize_graph(
                graph,
                title=f"{hospital} {split} Graph",
                save_path=img_path
            )

            print(f"Stored in {save_path}")

    print("All hospital graphs (train/val + test separately) created successfully.")


if __name__ == "__main__":
    main()