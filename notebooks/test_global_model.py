import torch
import os
import sys
import json
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from source.plot_global_results import (
    plot_global_model_performance,
    plot_federated_fairness_summary
)
sys.path.append(os.path.abspath(".."))
from data.models.gcn_model import GCN

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Testing on device: {device}")

# Hospital Testing Function
def test_on_hospital(global_model, hospital_name):
    graph_path = f"../data/graph/{hospital_name}_test.pt"
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
        labels,
        preds,
        target_names=['Not CKD', 'CKD'],
        output_dict=True,
        zero_division=0
    )
    num_samples = len(labels)
    return acc, report, num_samples

# Main Evaluation
def main():
    model_path = "../data/models/global_model.pth"
    hospitals = ["hospital_A", "hospital_B", "hospital_C"]

    if not os.path.exists(model_path):
        print(f"Error: Global model file not found at {model_path}.")
        return

    # Load model
    global_model = GCN(input_dim=24, hidden_dim=32, output_dim=2).to(device)

    try:
        global_model.load_state_dict(
            torch.load(model_path, map_location=device)
        )
        print("Global model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    print("\n" + "="*45)
    print("GLOBAL MODEL TEST SET EVALUATION")
    print("="*45)
    hospital_results = []
    for h in hospitals:
        source_name = (
            "UCI" if h == "hospital_A"
            else "Kaggle" if h == "hospital_B"
            else "Synthetic"
        )
        print(f"\nEvaluating {h} ({source_name})")
        result = test_on_hospital(global_model, h)
        if result:
            acc, report, samples = result
            hospital_results.append({
                "hospital": h,
                "accuracy": acc,
                "samples": samples
            })

            print(f"Accuracy : {acc:.4f}")
            print(f"Precision: {report['CKD']['precision']:.4f}")
            print(f"Recall   : {report['CKD']['recall']:.4f}")
            print(f"F1-score : {report['CKD']['f1-score']:.4f}")

    # Federated Summary
    if hospital_results:
        accuracies = [h["accuracy"] for h in hospital_results]
        samples = [h["samples"] for h in hospital_results]
        macro_acc = np.mean(accuracies)
        micro_acc = np.sum(
            [a * n for a, n in zip(accuracies, samples)]
        ) / np.sum(samples)
        std_acc = np.std(accuracies)
        print("\n" + "="*45)
        print("FEDERATED PERFORMANCE SUMMARY")
        print("="*45)
        print(f"Macro Accuracy (Fairness)  : {macro_acc:.4f}")
        print(f"Micro Accuracy (Deployment): {micro_acc:.4f}")
        print(f"Accuracy Std Dev           : {std_acc:.4f}")
        print("="*45)
        # Save results
        results = {
            "hospitals": hospital_results,
            "macro_accuracy": macro_acc,
            "micro_accuracy": micro_acc,
            "std_accuracy": std_acc
        }
        os.makedirs("../data/results", exist_ok=True)
        save_path = "../data/results/global_test_results.json"
        with open(save_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved → {save_path}")
        # Plot figures
        plot_global_model_performance()
        plot_federated_fairness_summary()
if __name__ == "__main__":
    main()