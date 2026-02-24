import torch
import torch.nn.functional as F
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import random
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from source.plot_resource_efficiency import plot_resource_efficiency

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Automatic import handling for ResourceMonitor
try:
    from notebooks.resource_monitor import ResourceMonitor
except (ImportError, ModuleNotFoundError):
    try:
        from resource_monitor import ResourceMonitor
    except (ImportError, ModuleNotFoundError):
        print("Warning: resource_monitor.py not found in expected paths.")

# Ensure pathing for GCN model import
sys.path.append(os.path.abspath(".."))
from data.models.gcn_model import GCN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_accuracy(model, graph):
    model.eval()
    with torch.no_grad():
        out = model(graph.x, graph.edge_index)
        preds = out.argmax(dim=1).cpu()
        labels = graph.y.cpu()
    return accuracy_score(labels, preds)


def train_local_model(train_graph_path, val_graph_path, epochs=60, lr=0.01, hospital_name="Hospital"):

    if not os.path.exists(train_graph_path) or not os.path.exists(val_graph_path):
        print(f"Error: Graph files for {hospital_name} not found.")
        return None, None, None

    train_graph = torch.load(train_graph_path, weights_only=False).to(device)
    val_graph = torch.load(val_graph_path, weights_only=False).to(device)

    torch.manual_seed(42)
    model = GCN(
        input_dim=train_graph.num_node_features,
        hidden_dim=32,
        output_dim=2
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # CLASS IMBALANCE 
    class_counts = torch.bincount(train_graph.y)
    class_counts = class_counts.float()

    # Avoid division by zero
    class_counts[class_counts == 0] = 1.0

    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    train_acc_list = []
    val_acc_list = []

    # ===== RESOURCE MONITOR =====
    monitor = ResourceMonitor()
    monitor.start_timer()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        out = model(train_graph.x, train_graph.edge_index)
        loss = criterion(out, train_graph.y)
        loss.backward()
        optimizer.step()

        train_acc = compute_accuracy(model, train_graph)
        val_acc = compute_accuracy(model, val_graph)

        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        print(f"[{hospital_name}] Epoch {epoch+1}/{epochs} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    # ===== RESOURCE REPORT =====
    training_time = monitor.stop_timer()
    memory_used = monitor.memory_usage_mb()

    print(f"\n[{hospital_name}] RESOURCE USAGE")
    print(f"Training Time : {training_time:.2f} seconds")
    print(f"Memory Usage  : {memory_used:.2f} MB")

    return model, train_acc_list, val_acc_list, training_time, memory_used


def plot_accuracy(train_acc, val_acc, hospital_name):
    plt.figure()
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"{hospital_name} - Train vs Val Accuracy")
    plt.legend()
    plt.grid(True)

    save_path = f"../data/graph/{hospital_name}_accuracy.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved accuracy plot: {save_path}")


def main():
    graph_dir = "../data/graph"
    model_save_dir = "../data/models"
    os.makedirs(model_save_dir, exist_ok=True)

    hospitals = ['A', 'B', 'C']
    resource_metrics = {}

    for h in hospitals:
        train_graph_path = os.path.join(graph_dir, f"hospital_{h}_train.pt")
        val_graph_path = os.path.join(graph_dir, f"hospital_{h}_val.pt")

        print(f"\n--- Training Hospital {h} ---")

        model, train_acc, val_acc, training_time, memory_used = train_local_model(
            train_graph_path,
            val_graph_path,
            hospital_name=f"Hospital {h}"
        )

        if model:
            save_path = os.path.join(model_save_dir, f"model_{h}.pth")
            torch.save(model.state_dict(), save_path)

            size_mb = ResourceMonitor.model_size_mb(save_path)
            print(f"Model Communication Cost: {size_mb:.2f} MB")
            resource_metrics[f"Hospital {h}"] = {
                "time": training_time,
                "memory": memory_used
            }

            plot_accuracy(train_acc, val_acc, f"hospital_{h}")
    plot_resource_efficiency(resource_metrics)

    print("\nLocal models trained and graphs generated successfully.")


if __name__ == "__main__":
    main()