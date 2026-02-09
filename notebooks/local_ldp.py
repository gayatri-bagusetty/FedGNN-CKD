import os
import sys
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Ensure the models directory is in the path for importing GCN
sys.path.append(os.path.abspath(".."))
from data.models.gcn_model import GCN

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
        # sigma = sensitivity / epsilon. Here alpha acts as sensitivity.
        noise_std = alpha / epsilon
        noise = torch.normal(
            mean=0.0,
            std=noise_std,
            size=param.data.size()
        ).to(device)

        param.data += noise
    return noisy_model

def select_best_epsilon(clean_model, graph, epsilons, alpha_score=0.8, hospital_name="Hospital"):
    """
    Iterates through epsilon values to find the best balance 
    between privacy and model accuracy.
    """
    clean_acc = evaluate(clean_model, graph)
    print(f"\n--- {hospital_name} LDP Optimization ---")
    print(f"Original Local Accuracy: {clean_acc:.4f}")

    best_score = -float("inf")
    best_eps = None
    best_model = None
    noisy_accuracies = []
    eps_max = max(epsilons)

    for eps in epsilons:
        # We apply LDP on the model weights
        noisy_model = apply_ldp(clean_model, eps)
        noisy_acc = evaluate(noisy_model, graph)

        # Optimization Score: High accuracy is good, high epsilon (less privacy) is bad
        # This formula helps pick the epsilon that protects data without killing accuracy
        score = alpha_score * noisy_acc - (1 - alpha_score) * (eps / eps_max)
        noisy_accuracies.append(noisy_acc)

        print(f"ε={eps:<3} | Protected Acc={noisy_acc:.4f} | Score={score:.4f}")

        if score > best_score:
            best_score = score
            best_eps = eps
            best_model = noisy_model

    print(f"Result: Selected optimal ε={best_eps} for {hospital_name}")
    return best_model, best_eps, noisy_accuracies, clean_acc

def plot_tradeoff(epsilons, accuracies, clean_acc, hospital_label):
    """Visualizes the impact of privacy noise on model accuracy."""
    plt.figure(figsize=(8, 5))
    plt.plot(epsilons, accuracies, marker="o", color="#1f77b4", linewidth=2, label="LDP Protected Accuracy")
    plt.axhline(y=clean_acc, color="#d62728", linestyle="--", label="Original Accuracy")
    plt.xlabel("Privacy Budget (Epsilon ε)")
    plt.ylabel("Accuracy")
    plt.title(f"Privacy–Utility Tradeoff: {hospital_label}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # Save plot for documentation
    save_path = f"../data/models/tradeoff_{hospital_label.replace(' ', '_')}.png"
    plt.savefig(save_path)
    plt.close() # Close to prevent display overlap in pipeline

def main():
    eps_values = [0.5, 1, 1.5, 2, 2.5, 3]
    hospitals = ['A', 'B', 'C']
    graph_dir = "../data/graph"
    model_dir = "../data/models"

    for h in hospitals:
        # 1. Load Data and Model
        graph_path = os.path.join(graph_dir, f"graph_{h}.pt")
        graph = load_graph(graph_path)
        
        # Initialize model with the graph's specific feature count (should be 24)
        model = GCN(graph.num_node_features, 32, 2).to(device)
        model_load_path = os.path.join(model_dir, f"model_{h}.pth")
        
        if not os.path.exists(model_load_path):
            print(f"Skipping Hospital {h}: Local model not found.")
            continue
            
        model.load_state_dict(torch.load(model_load_path, weights_only=True))

        # 2. Optimize Epsilon for this specific hospital data distribution
        best_ldp_model, best_eps, accs, clean_acc = select_best_epsilon(
            model, graph, eps_values, hospital_name=f"Hospital {h}"
        )

        # 3. Save the Protected Model for Federated Aggregation
        save_path = os.path.join(model_dir, f"model_{h}_ldp.pth")
        torch.save(best_ldp_model.state_dict(), save_path)
        
        # 4. Generate and save the visualization
        plot_tradeoff(eps_values, accs, clean_acc, f"Hospital {h}")

    print("\n[LDP Phase Complete] All protected local models saved to ../data/models/")

if __name__ == "__main__":
    main()