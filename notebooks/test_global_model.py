import torch
import os
import sys
import json
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from source.plot_global_results import plot_global_model_performance, plot_federated_fairness_summary

# Project path to import GCN model architecture
sys.path.append(os.path.abspath(".."))
from data.models.gcn_model import GCN

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Testing on device: {device}")

def test_on_hospital(global_model, hospital_name):
    """
    Loads a hospital's TEST graph and tests the global model against it.
    """
    graph_path = f"../data/graph/test/{hospital_name}_test.pt"

    if not os.path.exists(graph_path):
        print(f"Warning: Test graph for {hospital_name} not found.")
        return None

    graph = torch.load(graph_path, weights_only=False).to(device)

    global_model.eval()
    with torch.no_grad():
        out = global_model(graph.x, graph.edge_index)
        preds = out.argmax(dim=1).cpu().numpy()
        labels = graph.y.cpu().numpy()

    acc = accuracy_score(labels, preds)
    report = classification_report(
        labels, preds,
        target_names=['Not CKD', 'CKD'],
        output_dict=True,
        zero_division=0
    )

    num_samples = len(labels)
    return acc, report, num_samples


def main():
    model_path = "../data/models/global_model.pth"
    hospitals = ["hospital_A", "hospital_B", "hospital_C"]

    if not os.path.exists(model_path):
        print(f"Error: Global model file not found at {model_path}.")
        return

    # Initialize Global Model
    global_model = GCN(input_dim=24, hidden_dim=32, output_dim=2).to(device)

    try:
        global_model.load_state_dict(torch.load(model_path, map_location=device))
        print(">>> Global model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("\n" + "="*40)
    print("GLOBAL MODEL TEST SET EVALUATION")
    print("="*40)

    hospital_results = []

    for h in hospitals:
        source_name = "UCI" if h == "hospital_A" else "Kaggle" if h == "hospital_B" else "Synthetic"
        print(f"\nEvaluating {h} ({source_name}) on TEST data...")

        eval_result = test_on_hospital(global_model, h)
        if eval_result:
            acc, report, samples = eval_result
            hospital_results.append({
                "hospital": h,
                "accuracy": acc,
                "samples": samples
            })

            print(f"  Accuracy:  {acc:.4f}")
            print(f"  Precision: {report['CKD']['precision']:.4f}")
            print(f"  Recall:    {report['CKD']['recall']:.4f}")
            print(f"  F1-Score:  {report['CKD']['f1-score']:.4f}")

    # ===== Federated Summary =====
    if hospital_results:
        accuracies = [h["accuracy"] for h in hospital_results]
        samples = [h["samples"] for h in hospital_results]

        # Macro Average (Fairness)
        macro_acc = np.mean(accuracies)

        # Micro Average (Data-weighted / Deployment)
        micro_acc = np.sum(
            [a * n for a, n in zip(accuracies, samples)]
        ) / np.sum(samples)

        print("\n" + "="*40)
        print("FEDERATED PERFORMANCE SUMMARY (TEST DATA)")
        print("="*40)
        print(f"Macro Federated Accuracy (Fairness):     {macro_acc:.4f}")
        print(f"Micro Federated Accuracy (Deployment):   {micro_acc:.4f}")
        print("="*40)

        # ===== Save results for plotting =====
        results = {
            "hospitals": hospital_results,
            "macro_accuracy": macro_acc,
            "micro_accuracy": micro_acc
        }

        os.makedirs("../data/results", exist_ok=True)
        save_path = "../data/results/global_test_results.json"

        with open(save_path, "w") as f:
            json.dump(results, f, indent=4)

        print(f"\n[Saved] Global test results written to {save_path}")
    plot_global_model_performance()
    plot_federated_fairness_summary()

    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n[Saved] Global test results written to {save_path}")


if __name__ == "__main__":
    main()