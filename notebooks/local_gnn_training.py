import torch
import torch.nn.functional as F
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

# ===== PROJECT PATH SETUP =====
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from data.models.gcn_model import GCN
from source.plot_resource_efficiency import plot_resource_efficiency

# Optional resource monitor
try:
    from notebooks.resource_monitor import ResourceMonitor
except:
    from resource_monitor import ResourceMonitor

# ===== REPRODUCIBILITY =====
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# METRIC FUNCTION
def compute_metrics(model, graph):
    model.eval()
    with torch.no_grad():
        out = model(graph.x, graph.edge_index)
        preds = out.argmax(dim=1).cpu()
        labels = graph.y.cpu()

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")

    return acc, macro_f1

# LOCAL TRAINING FUNCTION
def train_local_model(train_graph_path, val_graph_path,
                      epochs=60, lr=0.01, hospital_name="Hospital"):

    train_graph = torch.load(train_graph_path, weights_only=False).to(device)
    val_graph = torch.load(val_graph_path, weights_only=False).to(device)

    # ===== MODEL (5 CLASS) =====
    model = GCN(
        input_dim=train_graph.num_node_features,
        hidden_dim=64,
        output_dim=5
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    # ===== CLASS WEIGHTS (IMBALANCE FIX) =====
    class_counts = torch.bincount(train_graph.y).float()
    class_counts[class_counts == 0] = 1.0
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    train_acc_list = []
    val_acc_list = []
    train_f1_list = []
    val_f1_list = []

    # ===== EARLY STOPPING =====
    best_val_loss = float("inf")
    best_model_state = None
    patience = 7
    patience_counter = 0

    monitor = ResourceMonitor()
    monitor.start_timer()

    for epoch in range(epochs):

        # ---- TRAIN ----
        model.train()
        optimizer.zero_grad()

        out = model(train_graph.x, train_graph.edge_index)
        train_loss = criterion(out, train_graph.y)
        train_loss.backward()
        optimizer.step()

        # ---- VALIDATION LOSS ----
        model.eval()
        with torch.no_grad():
            val_out = model(val_graph.x, val_graph.edge_index)
            val_loss = criterion(val_out, val_graph.y)

        train_acc, train_f1 = compute_metrics(model, train_graph)
        val_acc, val_f1 = compute_metrics(model, val_graph)

        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        train_f1_list.append(train_f1)
        val_f1_list.append(val_f1)

        print(f"[{hospital_name}] Epoch {epoch+1}/{epochs}")
        print(f"Train Acc: {train_acc:.4f} | Train MacroF1: {train_f1:.4f}")
        print(f"Val   Acc: {val_acc:.4f} | Val   MacroF1: {val_f1:.4f}")
        print("-" * 50)

        # ---- EARLY STOPPING LOGIC ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"{hospital_name} Early stopping triggered.")
            break

    # ---- RESTORE BEST MODEL ----
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    training_time = monitor.stop_timer()
    memory_used = monitor.memory_usage_mb()

    print(f"\n[{hospital_name}] RESOURCE USAGE")
    print(f"Training Time : {training_time:.2f} seconds")
    print(f"Memory Usage  : {memory_used:.2f} MB")

    return model, train_acc_list, val_acc_list, train_f1_list, val_f1_list, training_time, memory_used

# PLOT FUNCTION
def plot_metrics(train_acc, val_acc, train_f1, val_f1, hospital_name):

    plt.figure()
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.title(f"{hospital_name} Accuracy")
    plt.savefig(f"../data/plots/{hospital_name}_accuracy.png")
    plt.close()

    plt.figure()
    plt.plot(train_f1, label="Train Macro F1")
    plt.plot(val_f1, label="Validation Macro F1")
    plt.xlabel("Epochs")
    plt.ylabel("Macro F1")
    plt.legend()
    plt.grid(True)
    plt.title(f"{hospital_name} Macro F1")
    plt.savefig(f"../data/plots/{hospital_name}_macro_f1.png")
    plt.close()

# MAIN FUNCTION
def main():

    graph_dir = "../data/graph"
    model_save_dir = "../data/models"
    os.makedirs(model_save_dir, exist_ok=True)

    hospitals = ['A', 'B', 'C']
    resource_metrics = {}

    for h in hospitals:

        print(f"\n====== Training Hospital {h} ======")

        train_graph_path = os.path.join(graph_dir, f"hospital_{h}_train.pt")
        val_graph_path = os.path.join(graph_dir, f"hospital_{h}_val.pt")

        model, train_acc, val_acc, train_f1, val_f1, training_time, memory_used = train_local_model(
            train_graph_path,
            val_graph_path,
            hospital_name=f"Hospital {h}"
        )

        save_path = os.path.join(model_save_dir, f"model_{h}.pth")
        torch.save(model.state_dict(), save_path)

        size_mb = ResourceMonitor.model_size_mb(save_path)

        print(f"Model Size (Communication Cost): {size_mb:.2f} MB")

        resource_metrics[f"Hospital {h}"] = {
            "time": training_time,
            "memory": memory_used
        }

        plot_metrics(train_acc, val_acc, train_f1, val_f1, f"hospital_{h}")

    plot_resource_efficiency(resource_metrics)

    print("\nAll Local Models Trained Successfully.")


if __name__ == "__main__":
    main()