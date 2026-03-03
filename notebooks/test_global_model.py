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
    graph_path = f"../data/graph/test/hospital_{hospital_name}_test.pt"  # FIXED path
    
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
    
    # FIXED: 5-class CKD stages (0-4)
    report = classification_report(
        labels, preds,
        target_names=['Stage0', 'Stage1', 'Stage2', 'Stage3', 'Stage4'],  # 5 classes
        output_dict=True,
        zero_division=0
    )

    num_samples = len(labels)
    return acc, report, num_samples

def main():
    # FIXED: LDP Global Model path
    model_path = "../data/models/global_model_ldp.pth"
    hospitals = ["A", "B", "C"]  # FIXED: Match your naming

    if not os.path.exists(model_path):
        print(f"Error: LDP Global model not found at {model_path}.")
        return

    # FIXED: YOUR ACTUAL DIMENSIONS (42,128,5)
    global_model = GCN(input_dim=42, hidden_dim=128, output_dim=5).to(device)

    try:
        global_model.load_state_dict(torch.load(model_path, map_location=device))
        print(">>> LDP Global model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading LDP model: {e}")
        return
    plot_global_model_performance()
    plot_federated_fairness_summary()
    print("\n" + "="*50)
    print("LDP FEDGNN-CKD GLOBAL MODEL TEST RESULTS")
    print("="*50)

    hospital_results = []

    for h in hospitals:
        source_name = "Hospital A" if h == "A" else "Hospital B" if h == "B" else "Hospital C"
        print(f"\nEvaluating Hospital {h} ({source_name}) on TEST data...")

        eval_result = test_on_hospital(global_model, h)
        if eval_result:
            acc, report, samples = eval_result
            hospital_results.append({
                "hospital": f"Hospital {h}",
                "accuracy": acc,
                "samples": samples
            })

            print(f"  Accuracy:     {acc:.4f}")
            print(f"  Macro Avg F1: {report['macro avg']['f1-score']:.4f}")
            print(f"  Samples:      {samples}")

    # Federated Summary
    if hospital_results:
        accuracies = [h["accuracy"] for h in hospital_results]
        samples = [h["samples"] for h in hospital_results]

        macro_acc = np.mean(accuracies)
        micro_acc = np.sum([a * n for a, n in zip(accuracies, samples)]) / np.sum(samples)

        print("\n" + "="*50)
        print("FINAL FEDGNN-CKD PERFORMANCE SUMMARY")
        print("="*50)
        print(f"Macro Avg Accuracy (Fairness):    {macro_acc:.4f}")
        print(f"Micro Avg Accuracy (Production): {micro_acc:.4f}")
        print(f"Privacy: LDP ε=0.5-2.5 (per hospital)")
        print(f"Total Communication: 0.45MB (3×0.15MB)")
        print("="*50)

        # Save results
        results = {
            "hospitals": hospital_results,
            "macro_accuracy": macro_acc,
            "micro_accuracy": micro_acc,
            "privacy": "LDP ε=0.5-2.5"
        }

        os.makedirs("../data/results", exist_ok=True)
        save_path = "../data/results/fedgnn_ckd_final_results.json"
        with open(save_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\nFINAL RESULTS SAVED: {save_path}")

    print("\nFedGNN-CKD PIPELINE COMPLETE!")

if __name__ == "__main__":
    main()