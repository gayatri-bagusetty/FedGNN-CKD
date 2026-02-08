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

import torch
import os
from data.models.gcn_model import GCN


def update_fedavg(local_model_path, global_model_path):
    """
    Aggregate local model into global model using
    dashboard-safe incremental weighted FedAvg
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ⚠️ MUST MATCH train_local_model architecture
    input_dim = 24
    hidden_dim = 32
    output_dim = 2

    # Load local model
    local_model = GCN(input_dim, hidden_dim, output_dim).to(device)
    local_model.load_state_dict(torch.load(local_model_path, map_location=device))

    # Load or initialize global model
    global_model = GCN(input_dim, hidden_dim, output_dim).to(device)

    if os.path.exists(global_model_path):
        global_model.load_state_dict(torch.load(global_model_path, map_location=device))

    # -------- Dashboard-safe weighted FedAvg update --------
    global_state = global_model.state_dict()
    local_state = local_model.state_dict()

    alpha = 0.8   # trust existing global model
    beta = 0.2    # trust incoming hospital update

    for key in global_state:
        global_state[key] = alpha * global_state[key] + beta * local_state[key]

    global_model.load_state_dict(global_state)

    # Save updated global model
    torch.save(global_model.state_dict(), global_model_path)
    print("✅ FedAvg aggregation completed")


def fedavg(ldp_models):
    """
    Performs Federated Averaging (FedAvg) on LDP-protected weights.
    Aggregates Hospital A (UCI), B (Kaggle), and C (Synthetic).
    """

    global_model = copy.deepcopy(ldp_models[0])
    global_state = global_model.state_dict()

    for key in global_state.keys():
        stacked_params = torch.stack(
            [model.state_dict()[key].float() for model in ldp_models]
        )
        global_state[key] = torch.mean(stacked_params, dim=0)

    global_model.load_state_dict(global_state)
    return global_model


def main():
    model_dir = "../data/models"
    graph_ref_path = "../data/graph/graph_A.pt"

    if not os.path.exists(graph_ref_path):
        print("Error: Reference graph missing. Run graph_construction.py first.")
        return

    # Load reference graph for input dimension
    ref_graph = torch.load(graph_ref_path, weights_only=False)
    input_dim = ref_graph.num_node_features

    # Initialize hospital models
    model_A = GCN(input_dim, 32, 2).to(device)
    model_B = GCN(input_dim, 32, 2).to(device)
    model_C = GCN(input_dim, 32, 2).to(device)

    # Load LDP-protected weights
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

    # Save global model
    save_path = os.path.join(model_dir, "global_model.pth")
    torch.save(global_model.state_dict(), save_path)
    print(f"Global Model successfully stored at: {save_path}")


if __name__ == "__main__":
    main()