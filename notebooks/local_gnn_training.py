import torch
import os
from sklearn.metrics import accuracy_score
import sys
import os

# Add project 'data' folder to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from models.gcn_model import GCN

# -------------------------------
# DEVICE
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# EVALUATION
# -------------------------------
def evaluate_model(model, graph):
    """Evaluate a model on a PyG graph and return accuracy."""
    model.eval()
    graph = graph.to(device)
    with torch.no_grad():
        logits = model(graph.x, graph.edge_index)
        preds = logits.argmax(dim=1).cpu()
        labels = graph.y.cpu()
    return accuracy_score(labels, preds)


# -------------------------------
# TRAIN SINGLE GRAPH
# -------------------------------
def train_single_graph(graph, global_model=None, epochs=50, lr=0.01, hidden_dim=32):
    """
    Train a local GCN on one graph.
    If a global_model is provided, use it as initial weights.
    """
    input_dim = graph.num_node_features
    output_dim = 2

    # Use provided global model or create a new one
    if global_model is not None:
        model = global_model.to(device)
    else:
        model = GCN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(graph.x.to(device), graph.edge_index.to(device))
        loss = criterion(out, graph.y.to(device))
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    acc = evaluate_model(model, graph)
    return model, acc


# -------------------------------
# LEGACY WRAPPER
# -------------------------------
def train_with_global_weights(graph, model=None, epochs=50, lr=0.01, hidden_dim=32):
    """Wrapper for admin/dashboard compatibility."""
    return train_single_graph(graph, global_model=model, epochs=epochs, lr=lr, hidden_dim=hidden_dim)


# -------------------------------
# LOCAL TRAINING PIPELINE
# -------------------------------
def train_local_pipeline(new_graph=None, global_model=None, epochs=50, lr=0.01, hidden_dim=32):
    """
    ADMIN PIPELINE PATTERN:
    - If new_graph + global_model provided → train only that graph
    - Else → load default hospital graphs (A, B, C) and train them
    """
    trained_models = {}

    # ---- ADMIN PIPELINE MODE ----
    if new_graph is not None:
        print("\n--- Training using provided graph ---")
        model, acc = train_single_graph(
            new_graph,
            global_model=global_model,
            epochs=epochs,
            lr=lr,
            hidden_dim=hidden_dim
        )
        trained_models["new_graph"] = {"model": model, "accuracy": acc}
        print(f"Local accuracy: {acc:.4f}")
        return trained_models

    # ---- DEFAULT HOSPITAL MODE ----
    print("\n--- Training default hospital graphs ---")
    graph_dir = "../data/graph"
    hospitals = ["A", "B", "C"]

    for h in hospitals:
        graph_path = os.path.join(graph_dir, f"graph_{h}.pt")
        if not os.path.exists(graph_path):
            print(f"Graph not found: {graph_path}")
            continue

        print(f"\nHospital {h}")
        graph = torch.load(graph_path, weights_only=False)

        model, acc = train_single_graph(
            graph,
            global_model=global_model,
            epochs=epochs,
            lr=lr,
            hidden_dim=hidden_dim
        )

        trained_models[h] = {"model": model, "accuracy": acc}
        print(f"Hospital {h} accuracy: {acc:.4f}")

    return trained_models


# -------------------------------
# TEST
# -------------------------------
if __name__ == "__main__":
    results = train_local_pipeline()
    print("Local training completed.")