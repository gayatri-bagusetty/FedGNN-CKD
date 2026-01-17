import torch
import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# Project path to import GCN model architecture
sys.path.append(os.path.abspath("../data"))
from models.gcn_model import GCN

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Testing on device: {device}")

def test_global_model(global_model, graph):
    """
    Tests the global model on a specific hospital's graph.
    Returns accuracy and detailed metrics.
    """
    global_model.eval()
    with torch.no_grad():
        # Ensure data is on the same device as the model
        x = graph.x.to(device)
        edge_index = graph.edge_index.to(device)
        
        # Forward pass
        out = global_model(x, edge_index)
        preds = out.argmax(dim=1).cpu().numpy()
        labels = graph.y.cpu().numpy()

    acc = accuracy_score(labels, preds)
    report = classification_report(labels, preds, target_names=['Not CKD', 'CKD'], output_dict=True)
    return acc, report

def main():
    # 1. Define Paths
    graph_path = "../data/graph/graph_A.pt"
    model_path = "../data/models/global_model.pth"

    if not os.path.exists(graph_path) or not os.path.exists(model_path):
        print("Error: Required graph or global model file not found.")
        return

    # 2. Load the evaluation graph
    print(f"Loading evaluation graph from {graph_path}...")
    graph = torch.load(graph_path, weights_only=False)
    input_dim = graph.num_node_features

    # 3. Load Global Model
    global_model = GCN(input_dim, 32, 2).to(device)
    try:
        global_model.load_state_dict(
            torch.load(model_path, map_location=device)
        )
        print("Global model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 4. Perform Evaluation
    print("\n--- Running Evaluation ---")
    accuracy, report = test_global_model(global_model, graph)

    # 5. Output Results
    print(f"Global Model Accuracy: {accuracy:.4f}")
    print("\nDetailed Performance:")
    print(f"  - Precision (CKD): {report['CKD']['precision']:.4f}")
    print(f"  - Recall (CKD):    {report['CKD']['recall']:.4f}")
    print(f"  - F1-Score (CKD):  {report['CKD']['f1-score']:.4f}")

if __name__ == "__main__":
    main()