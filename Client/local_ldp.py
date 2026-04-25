import os
import sys
import copy
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Ensure the models directory is in the path for importing GCN
sys.path.append(os.path.abspath(".."))
from data.models.gcn_model import GCN

# GLOBAL SEED FOR REPRODUCIBILITY
SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
GRAPH_DIR = "../data/graph"
MODEL_DIR = "../data/models"
os.makedirs(MODEL_DIR, exist_ok=True)

# LOAD GRAPH
def load_graph(path):
    """Loads the PyTorch Geometric graph to the active device."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph not found at {path}")
    return torch.load(path, weights_only=False).to(device)

# MODEL EVALUATION
def evaluate(model, graph):
    """Calculates accuracy for a GCN model."""
    model.eval()
    with torch.no_grad():
        out = model(graph.x, graph.edge_index)
        preds = out.argmax(dim=1).cpu()
        labels = graph.y.cpu()
    return accuracy_score(labels, preds)

# APPLY LOCAL DIFFERENTIAL PRIVACY
def apply_ldp(model, epsilon, alpha=0.3):
    """
    Applies Local Differential Privacy by adding Gaussian noise
    to model parameters.
    """
    noisy_model = copy.deepcopy(model)
    noise_std = alpha / epsilon
    for name, param in noisy_model.named_parameters():
        if not param.requires_grad:
            continue
        noise = torch.normal(
            mean=0.0,
            std=noise_std,
            size=param.data.size(),
            device=device
        )
        param.data += noise
    return noisy_model

# EPSILON SELECTION
def select_best_epsilon(clean_model, graph, epsilons, delta=0.02, hospital_name="Hospital"):
    clean_acc = evaluate(clean_model, graph)
    print(f"\n--- {hospital_name} LDP Optimization ---")
    print(f"Original Local Accuracy: {clean_acc:.4f}")
    threshold = clean_acc - delta
    print(f"Minimum Acceptable Accuracy: {threshold:.4f}")
    best_eps = None
    best_model = None
    noisy_accuracies = []
    for eps in sorted(epsilons):
        noisy_model = apply_ldp(clean_model, eps)
        noisy_acc = evaluate(noisy_model, graph)
        noisy_accuracies.append(noisy_acc)
        print(f"ε={eps:<3} | Protected Acc={noisy_acc:.4f}")
        if noisy_acc >= threshold and best_eps is None:
            best_eps = eps
            best_model = noisy_model
    # fallback if none meet threshold
    if best_eps is None:
        print("No epsilon satisfies constraint. Selecting best accuracy.")
        best_idx = np.argmax(noisy_accuracies)
        best_eps = epsilons[best_idx]
        best_model = apply_ldp(clean_model, best_eps)
    print(f"Selected optimal ε={best_eps} for {hospital_name}")
    return best_model, best_eps, noisy_accuracies, clean_acc

# TRADEOFF PLOT
def plot_tradeoff(epsilons, accuracies, clean_acc, hospital_label):
    plt.figure(figsize=(8,5))
    plt.plot(epsilons, accuracies, marker="o", linewidth=2,
             label="LDP Protected Accuracy")
    plt.axhline(y=clean_acc, linestyle="--",
                label="Original Accuracy")
    plt.xlabel("Privacy Budget (ε)")
    plt.ylabel("Accuracy")
    plt.title(f"Privacy–Utility Trade-off under LDP {hospital_label}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    save_path = os.path.join(
        MODEL_DIR,
        f"tradeoff_{hospital_label.replace(' ','_')}.png"
    )

    plt.savefig(save_path)
    plt.close()

# MAIN
def main():
    eps_values = [0.5, 1, 1.5, 2, 2.5, 3]
    hospitals = ['A', 'B', 'C']
    for h in hospitals:
        graph_path = os.path.join(GRAPH_DIR, f"hospital_{h}_train.pt")
        model_path = os.path.join(MODEL_DIR, f"model_{h}.pth")
        if not os.path.exists(graph_path):
            print(f"[SKIP] Graph missing for Hospital {h}: {graph_path}")
            continue
        if not os.path.exists(model_path):
            print(f"[SKIP] Model missing for Hospital {h}: {model_path}")
            continue
        graph = load_graph(graph_path)
        model = GCN(graph.num_node_features, 32, 2).to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True))

        best_ldp_model, best_eps, accs, clean_acc = select_best_epsilon(
            model,
            graph,
            eps_values,
            hospital_name=f"Hospital {h}"
        )
        save_model_path = os.path.join(MODEL_DIR, f"model_{h}_ldp.pth")
        torch.save(best_ldp_model.state_dict(), save_model_path)
        plot_tradeoff(eps_values, accs, clean_acc, f"Hospital {h}")
        print(f"[SAVED] LDP model → {save_model_path}")
    print("\n[LDP Phase Complete] All protected local models saved.")

if __name__ == "__main__":
    main()