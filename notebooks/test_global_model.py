import torch
import os
import sys
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------------------------
# PATH
# -------------------------------------------------
sys.path.append(os.path.abspath("../data"))
from models.gcn_model import GCN

# -------------------------------------------------
# DEVICE
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Testing on device: {device}")

# -------------------------------------------------
# TEST FUNCTION
# -------------------------------------------------
def test_global_model(global_model, graph):
    """
    Tests the global model on a specific graph.
    Returns accuracy and detailed metrics.
    """
    global_model.eval()
    graph = graph.to(device)

    with torch.no_grad():
        out = global_model(graph.x, graph.edge_index)
        preds = out.argmax(dim=1).cpu().numpy()
        labels = graph.y.cpu().numpy()

    acc = accuracy_score(labels, preds)
    report = classification_report(labels, preds, target_names=['Not CKD', 'CKD'], output_dict=True)
    return acc, report


# -------------------------------------------------
# ADMIN PIPELINE MODE FUNCTION
# -------------------------------------------------
def evaluate_global_model(global_model=None, graph=None):
    """
    ADMIN PIPELINE PATTERN

    If global_model and graph provided:
        → evaluate directly
    Else:
        → load default graph_A and global model from disk
    """

    # DEFAULT PATHS
    graph_path = "../data/graph/graph_A.pt"
    model_path = "../data/models/global_model.pth"

    # -------------------------------------------------
    # CASE 1 — PROVIDED
    # -------------------------------------------------
    if global_model is not None and graph is not None:
        print("\n--- Evaluating provided global model ---")
        return test_global_model(global_model, graph)

    # -------------------------------------------------
    # CASE 2 — DEFAULT FILES
    # -------------------------------------------------
    if not os.path.exists(graph_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Required default graph or global model not found.")

    print(f"\n--- Loading default graph from {graph_path} ---")
    graph = torch.load(graph_path, weights_only=False)
    input_dim = graph.num_node_features

    print(f"--- Loading default global model from {model_path} ---")
    global_model = GCN(input_dim, 32, 2).to(device)
    global_model.load_state_dict(torch.load(model_path, map_location=device))

    return test_global_model(global_model, graph)


# -------------------------------------------------
# TEST / CLI
# -------------------------------------------------
if __name__ == "__main__":
    acc, report = evaluate_global_model()
    print(f"Global Model Accuracy: {acc:.4f}")
    print("\nDetailed Performance:")
    print(f"  - Precision (CKD): {report['CKD']['precision']:.4f}")
    print(f"  - Recall (CKD):    {report['CKD']['recall']:.4f}")
    print(f"  - F1-Score (CKD):  {report['CKD']['f1-score']:.4f}")