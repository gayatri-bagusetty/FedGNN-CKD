import os
import sys
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Ensure the models directory is in the path for importing GCN
sys.path.append(os.path.abspath("../data"))
from models.gcn_model import GCN

# Check Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_graph(path):
    """Loads the PyTorch Geometric graph to the active device."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph not found at {path}")
    return torch.load(path, weights_only=False).to(device)

def evaluate(model, graph):
    """Calculates accuracy for a GCN model on a specific graph."""
    model.eval()
    with torch.no_grad():
        out = model(graph.x, graph.edge_index)
        preds = out.argmax(dim=1).cpu()
        labels = graph.y.cpu()
    return accuracy_score(labels, preds)

def apply_ldp(model, epsilon, alpha=0.3):
    """
    Applies Local Differential Privacy by adding Gaussian noise to model weights.
    Higher epsilon = Less noise (Less privacy, more utility).
    Lower epsilon = More noise (More privacy, less utility).
    """
    noisy_model = copy.deepcopy(model)
    for _, param in noisy_model.named_parameters():
        if not param.requires_grad:
            continue

        # Standard deviation for noise based on privacy budget
        noise_std = alpha / epsilon
        noise = torch.normal(
            mean=0.0,
            std=noise_std,
            size=param.data.size()
        ).to(device)

        param.data += noise
    return noisy_model

def select_best_epsilon(clean_model, graph, epsilons, alpha=0.8, hospital_name="Hospital"):
    """
    Iterates through epsilon values to find the best balance 
    between privacy and model accuracy.
    """
    clean_acc = evaluate(clean_model, graph)
    print(f"\n--- {hospital_name} Optimization ---")
    print(f"Clean Accuracy: {clean_acc:.4f}")

    best_score = -float("inf")
    best_eps = None
    best_model = None
    noisy_accuracies = []
    eps_max = max(epsilons)

    for eps in epsilons:
        noisy_model = apply_ldp(clean_model, eps)
        noisy_acc = evaluate(noisy_model, graph)

        # Optimization Score: High accuracy is good, high epsilon (less privacy) is bad
        score = alpha * noisy_acc - (1 - alpha) * (eps / eps_max)
        noisy_accuracies.append(noisy_acc)

        print(f"ε={eps:<3} | Noisy Acc={noisy_acc:.4f} | Score={score:.4f}")

        if score > best_score:
            best_score = score
            best_eps = eps
            best_model = noisy_model

    print(f"Result: Selected ε={best_eps} for {hospital_name}")
    return best_model, best_eps, noisy_accuracies, clean_acc

def plot_tradeoff(epsilons, accuracies, clean_acc, hospital_label):
    """Visualizes the impact of privacy noise on model accuracy."""
    plt.figure(figsize=(8, 5))
    plt.plot(epsilons, accuracies, marker="o", color="blue", label="LDP Protected Accuracy")
    plt.axhline(y=clean_acc, color="red", linestyle="--", label="Original Accuracy")
    plt.xlabel("Epsilon (ε) - Privacy Budget")
    plt.ylabel("Accuracy")
    plt.title(f"Privacy–Utility Tradeoff ({hospital_label})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()

def main():
    eps_values = [1.5, 2, 3, 4, 5]
    hospitals = ['A', 'B', 'C']
    graph_dir = "../data/graph"
    model_dir = "../data/models"

    for h in hospitals:
        # 1. Load Data and Model
        graph = load_graph(f"{graph_dir}/graph_{h}.pt")
        model = GCN(graph.num_node_features, 32, 2).to(device)
        model.load_state_dict(torch.load(f"{model_dir}/model_{h}.pth"))

        # 2. Optimize Epsilon
        best_ldp_model, best_eps, accs, clean_acc = select_best_epsilon(
            model, graph, eps_values, hospital_name=f"Hospital {h}"
        )

        # 3. Save the Optimized LDP Model
        save_path = f"{model_dir}/model_{h}_ldp.pth"
        torch.save(best_ldp_model.state_dict(), save_path)
        
        # 4. Show results
        plot_tradeoff(eps_values, accs, clean_acc, f"Hospital {h}")

    print("\nAll LDP-optimized models saved to ../data/models/")

if __name__ == "__main__":
    main()