import torch
import torch.nn.functional as F
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve
import random
import numpy as np

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

def compute_metrics(model, graph):
    model.eval()
    with torch.no_grad():
        out = model(graph.x, graph.edge_index)
        preds = out.argmax(dim=1).cpu().numpy()
        labels = graph.y.cpu().numpy()

    probs = F.softmax(out, dim=1)[:,1].cpu().numpy()
    roc_auc = roc_auc_score(labels, probs)
    fpr, tpr, _ = roc_curve(labels, probs)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    return roc_auc, precision, recall, f1, fpr, tpr

def plot_roc_curve(fpr, tpr, roc_auc, hospital_name):
    plt.figure(figsize=(6,5))

    plt.plot(fpr, tpr,
             linewidth=2,
             label=f"{hospital_name} (AUC = {roc_auc:.4f})")

    plt.plot([0,1], [0,1],
             linestyle='--',
             linewidth=1,
             label="Random Classifier")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{hospital_name} - ROC Curve")

    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    save_path = f"../data/plots/{hospital_name}_roc.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"{hospital_name} ROC-AUC: {roc_auc:.4f}")
    print(f"Saved ROC curve: {save_path}")
    
def train_local_model(train_graph_path, val_graph_path,
                      epochs=100, lr=0.01, hospital_name="Hospital"):

    if not os.path.exists(train_graph_path) or not os.path.exists(val_graph_path):
        print(f"Error: Graph files for {hospital_name} not found.")
        return None, None, None, None, None

    train_graph = torch.load(train_graph_path, weights_only=False).to(device)
    val_graph = torch.load(val_graph_path, weights_only=False).to(device)

    model = GCN(
        input_dim=train_graph.num_node_features,
        hidden_dim=32,
        output_dim=2
    ).to(device)

    # L2 regularization (weight decay)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    # ----- Class imbalance handling -----
    class_counts = torch.bincount(train_graph.y).float()
    class_counts[class_counts == 0] = 1.0
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    train_acc_list = []
    val_acc_list = []

    # ===== EARLY STOPPING SETUP =====
    best_val_loss = float("inf")
    best_model_state = None
    patience = 15
    patience_counter = 0

    # ===== RESOURCE MONITOR =====
    monitor = ResourceMonitor()
    monitor.start_timer()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        out = model(train_graph.x, train_graph.edge_index)
        train_loss = criterion(out, train_graph.y)
        train_loss.backward()
        optimizer.step()

        # ----- Validation loss -----
        model.eval()
        with torch.no_grad():
            val_out = model(val_graph.x, val_graph.edge_index)
            val_loss = criterion(val_out, val_graph.y)

        train_acc = compute_accuracy(model, train_graph)
        val_acc = compute_accuracy(model, val_graph)

        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        print(f"[{hospital_name}] Epoch {epoch+1}/{epochs} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # ===== EARLY STOPPING LOGIC =====
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"[{hospital_name}] Early stopping triggered at epoch {epoch+1}")
            break

    # ===== RESTORE BEST MODEL =====
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # ===== FINAL VALIDATION METRICS =====
    val_acc = compute_accuracy(model, val_graph)
    roc_auc, val_precision, val_recall, val_f1, fpr, tpr = compute_metrics(model, val_graph)

    print(f"\n[{hospital_name}] Validation Metrics")
    print(f"Accuracy : {val_acc:.4f}")
    print(f"Precision: {val_precision:.4f}")
    print(f"Recall   : {val_recall:.4f}")
    print(f"F1-score : {val_f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4F}")
    
    plot_roc_curve(fpr, tpr, roc_auc, hospital_name)
    # ===== RESOURCE REPORT =====
    training_time = monitor.stop_timer()
    memory_used = monitor.memory_usage_mb()

    print(f"\n[{hospital_name}] RESOURCE USAGE")
    print(f"Training Time : {training_time:.2f} seconds")
    print(f"Memory Usage  : {memory_used:.2f} MB")

    return model, train_acc_list, val_acc_list, training_time, memory_used, val_acc, val_precision, val_recall, val_f1, roc_auc

def plot_accuracy(train_acc, val_acc, hospital_name):
    plt.figure()
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"{hospital_name} - Train vs Val Accuracy")
    plt.legend()
    plt.grid(True)

    save_path = f"../data/plots/{hospital_name}_accuracy.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved accuracy plot: {save_path}")


def main():
    graph_dir = "../data/graph"
    model_save_dir = "../data/models"
    os.makedirs(model_save_dir, exist_ok=True)

    hospitals = ['A', 'B', 'C']
    resource_metrics = {}
    metrics_records = []
    os.makedirs("../data/file", exist_ok=True)

    for h in hospitals:
        train_graph_path = os.path.join(graph_dir, f"hospital_{h}_train.pt")
        val_graph_path = os.path.join(graph_dir, f"hospital_{h}_val.pt")

        print(f"\n--- Training Hospital {h} ---")

        model, train_acc, val_acc, training_time, memory_used, val_acc_final, precision, recall, f1, roc_auc = train_local_model(
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
            metrics_records.append({
                "hospital": f"Hospital {h}",
                "accuracy": val_acc_final,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc, 
                "training_time_sec": training_time,
                "memory_mb": memory_used
            })

            plot_accuracy(train_acc, val_acc, f"hospital_{h}")
    metrics_df = pd.DataFrame(metrics_records)
    metrics_path = "../data/file/local_training_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nLocal training metrics saved → {metrics_path}")
    plot_resource_efficiency(resource_metrics)

    print("\nLocal models trained and graphs generated successfully.")


if __name__ == "__main__":
    main()