import torch
import os
import copy
import sys

# Ensure the project structure is respected for imports
sys.path.append(os.path.abspath(".."))
from data.models.gcn_model import GCN

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"FedAvg Server active on: {device}")

def fedavg(ldp_models):
    """
    Performs Federated Averaging (FedAvg) on LDP-protected weights.
    Aggregates Hospital A (UCI), B (Kaggle), and C (Synthetic).
    """
    global_model = copy.deepcopy(ldp_models[0])
    global_state = global_model.state_dict()

    for key in global_state.keys():
        # Average the parameters across all three heterogeneous hospital models
        stacked_params = torch.stack([model.state_dict()[key].float() for model in ldp_models])
        global_state[key] = torch.mean(stacked_params, dim=0)

    global_model.load_state_dict(global_state)
    return global_model

def main():
    model_dir = "../data/models"
    # Use graph_A (UCI) as the reference for input dimensions (24 features)
    graph_ref_path = "../data/graph/graph_A.pt"

    if not os.path.exists(graph_ref_path):
        print("Error: Reference graph missing. Run graph_construction.py first.")
        return

    # Load reference to determine architecture dimensions
    ref_graph = torch.load(graph_ref_path, weights_only=False)
    input_dim = ref_graph.num_node_features
    
    # Initialize model containers
    model_A = GCN(input_dim, 32, 2).to(device)
    model_B = GCN(input_dim, 32, 2).to(device)
    model_C = GCN(input_dim, 32, 2).to(device)

    # Load LDP weights sent from the three hospitals
    try:
        model_A.load_state_dict(torch.load(f"{model_dir}/model_A_ldp.pth", weights_only=True))
        model_B.load_state_dict(torch.load(f"{model_dir}/model_B_ldp.pth", weights_only=True))
        model_C.load_state_dict(torch.load(f"{model_dir}/model_C_ldp.pth", weights_only=True))
        print(">>> Successfully received LDP-protected weights from all hospitals.")
    except Exception as e:
        print(f"Error loading hospital weights: {e}")
        return

    print(">>> Executing Federated Aggregation (FedAvg)...")
    global_model = fedavg([model_A, model_B, model_C])

    # Save the consolidated Global Model
    save_path = os.path.join(model_dir, "global_model.pth")
    torch.save(global_model.state_dict(), save_path)
    print(f"Global Model successfully stored at: {save_path}")

if __name__ == "__main__":
    main()