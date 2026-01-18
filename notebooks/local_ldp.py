import os
import sys
import copy
import torch
import numpy as np
from sklearn.metrics import accuracy_score

# -------------------------------------------------
# PATH SETUP
# -------------------------------------------------
sys.path.append(os.path.abspath("../data"))
from models.gcn_model import GCN

# -------------------------------------------------
# DEVICE
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------------------------------------
# ACCURACY
# -------------------------------------------------
def evaluate(model, graph):
    model.eval()
    graph = graph.to(device)

    with torch.no_grad():
        out = model(graph.x, graph.edge_index)
        preds = out.argmax(dim=1).cpu()
        labels = graph.y.cpu()

    return accuracy_score(labels, preds)


# -------------------------------------------------
# APPLY LDP
# -------------------------------------------------
def apply_ldp(model, epsilon, alpha=0.3):
    """
    Adds Gaussian noise to model weights
    """
    noisy_model = copy.deepcopy(model)

    for param in noisy_model.parameters():
        if not param.requires_grad:
            continue

        noise_std = alpha / epsilon
        noise = torch.normal(
            mean=0.0,
            std=noise_std,
            size=param.data.size()
        ).to(device)

        param.data += noise

    return noisy_model


# -------------------------------------------------
# EPSILON SELECTION
# -------------------------------------------------
def select_best_epsilon(model, graph, epsilons, alpha=0.8, label="Node"):
    clean_acc = evaluate(model, graph)

    print(f"\n--- {label} LDP Optimization ---")
    print(f"Clean accuracy: {clean_acc:.4f}")

    best_score = -1e9
    best_eps = None
    best_model = None
    acc_list = []

    eps_max = max(epsilons)

    for eps in epsilons:
        noisy_model = apply_ldp(model, eps)
        noisy_acc = evaluate(noisy_model, graph)

        score = alpha * noisy_acc - (1 - alpha) * (eps / eps_max)
        acc_list.append(noisy_acc)

        print(f"ε={eps:<4} | acc={noisy_acc:.4f} | score={score:.4f}")

        if score > best_score:
            best_score = score
            best_eps = eps
            best_model = noisy_model

    print(f"Selected ε = {best_eps}")

    return best_model, best_eps, acc_list, clean_acc


# -------------------------------------------------
# MAIN LDP PIPELINE
# -------------------------------------------------
def ldp_pipeline(
        trained_model=None,
        graph=None,
        eps_values=None
):
    """
    ADMIN PIPELINE PATTERN

    If trained_model + graph provided:
        → apply LDP
        → return noisy model + accuracy

    Else:
        → apply LDP on default hospital models
    """

    if eps_values is None:
        eps_values = [1.5, 2, 3, 4, 5]

    results = {}

    # =====================================================
    # CASE 1 — ADMIN PIPELINE MODE
    # =====================================================
    if trained_model is not None and graph is not None:

        print("\n--- Applying LDP to provided local model ---")

        best_model, best_eps, accs, clean_acc = select_best_epsilon(
            trained_model,
            graph,
            eps_values,
            label="New Graph"
        )

        final_acc = evaluate(best_model, graph)

        return {
            "model": best_model,
            "epsilon": best_eps,
            "accuracy": final_acc,
            "clean_accuracy": clean_acc
        }

    # =====================================================
    # CASE 2 — DEFAULT HOSPITAL MODE
    # =====================================================
    print("\n--- Applying LDP to hospital models ---")

    model_dir = "../data/models"
    graph_dir = "../data/graph"

    hospitals = ["A", "B", "C"]

    for h in hospitals:

        print(f"\nHospital {h}")

        graph_path = os.path.join(graph_dir, f"graph_{h}.pt")
        model_path = os.path.join(model_dir, f"model_{h}.pth")

        if not os.path.exists(graph_path) or not os.path.exists(model_path):
            print("Missing model or graph — skipping")
            continue

        graph = torch.load(graph_path, weights_only=False).to(device)

        model = GCN(
            graph.num_node_features,
            32,
            len(torch.unique(graph.y))
        ).to(device)

        model.load_state_dict(torch.load(model_path))

        best_model, best_eps, accs, clean_acc = select_best_epsilon(
            model,
            graph,
            eps_values,
            label=f"Hospital {h}"
        )

        final_acc = evaluate(best_model, graph)

        torch.save(
            best_model.state_dict(),
            os.path.join(model_dir, f"model_{h}_ldp.pth")
        )

        results[h] = {
            "model": best_model,
            "epsilon": best_eps,
            "accuracy": final_acc,
            "clean_accuracy": clean_acc
        }

    return results


# -------------------------------------------------
# TEST
# -------------------------------------------------
if __name__ == "__main__":
    results = ldp_pipeline()
    print("LDP completed.")