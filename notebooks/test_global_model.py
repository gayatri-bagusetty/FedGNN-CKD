import torch
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# Project path to import GCN model architecture
sys.path.append(os.path.abspath(".."))
from data.models.gcn_model import GCN

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Testing on device: {device}")

def test_on_hospital(global_model, hospital_id):
    """
    Loads a hospital's specific graph and tests the global model against it.
    """
    graph_path = f"../data/graph/graph_{hospital_id}.pt"
    if not os.path.exists(graph_path):
        print(f"Warning: Graph for Hospital {hospital_id} not found.")
        return None

    graph = torch.load(graph_path, weights_only=False).to(device)
    
    global_model.eval()
    with torch.no_grad():
        # Forward pass on the full graph of the hospital
        out = global_model(graph.x, graph.edge_index)
        preds = out.argmax(dim=1).cpu().numpy()
        labels = graph.y.cpu().numpy()

    acc = accuracy_score(labels, preds)
    # Output metrics as a dictionary
    report = classification_report(labels, preds, target_names=['Not CKD', 'CKD'], output_dict=True, zero_division=0)
    return acc, report

def main():
    model_path = "../data/models/global_model.pth"
    hospitals = ['A', 'B', 'C'] # UCI, Kaggle, Synthetic

    if not os.path.exists(model_path):
        print(f"Error: Global model file not found at {model_path}.")
        return

    # 1. Initialize Global Model (Architecture must match 24 features)
    # We assume 24 features based on the updated preprocessing
    global_model = GCN(input_dim=24, hidden_dim=32, output_dim=2).to(device)
    
    try:
        global_model.load_state_dict(torch.load(model_path, map_location=device))
        print(">>> Global model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("\n" + "="*40)
    print("GLOBAL MODEL MULTI-HOSPITAL EVALUATION")
    print("="*40)

    results = []

    # 2. Iterate through each hospital to test generalization
    for h in hospitals:
        source_name = "UCI" if h == 'A' else "Kaggle" if h == 'B' else "Synthetic"
        print(f"\nEvaluating Hospital {h} ({source_name})...")
        
        eval_result = test_on_hospital(global_model, h)
        
        if eval_result:
            acc, report = eval_result
            results.append(acc)
            print(f"  Accuracy:  {acc:.4f}")
            print(f"  Precision: {report['CKD']['precision']:.4f}")
            print(f"  Recall:    {report['CKD']['recall']:.4f}")
            print(f"  F1-Score:  {report['CKD']['f1-score']:.4f}")

    # 3. Final Federated Performance Summary
    if results:
        avg_acc = np.mean(results)
        print("\n" + "="*40)
        print(f"AVERAGE FEDERATED ACCURACY: {avg_acc:.4f}")
        print("="*40)

if __name__ == "__main__":
    main()