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

# INCREMENTAL UPDATE (Dashboard use)
def update_fedavg(local_model_path, global_model_path):
    """
    Incrementally update global model using weighted FedAvg
    """

    # MUST match hospital-side architecture
    input_dim = 42
    hidden_dim = 128
    output_dim = 5

    # Load local model
    local_model = GCN(input_dim, hidden_dim, output_dim).to(device)
    local_model.load_state_dict(torch.load(local_model_path, map_location=device))

    # Load or initialize global model
    global_model = GCN(input_dim, hidden_dim, output_dim).to(device)

    if os.path.exists(global_model_path):
        global_model.load_state_dict(torch.load(global_model_path, map_location=device))

    global_state = global_model.state_dict()
    local_state = local_model.state_dict()

    alpha = 0.8   # old global weight
    beta = 0.2    # new local weight

    for key in global_state:
        global_state[key] = alpha * global_state[key] + beta * local_state[key]

    global_model.load_state_dict(global_state)

    torch.save(global_model.state_dict(), global_model_path)
    print("FedAvg incremental update completed")


# FULL FEDAVG (Batch use)
def fedavg(models):
    """
    Performs standard FedAvg over multiple models
    """
    global_model = copy.deepcopy(models[0])
    global_state = global_model.state_dict()

    for key in global_state.keys():
        stacked_params = torch.stack(
            [model.state_dict()[key].float() for model in models]
        )
        global_state[key] = torch.mean(stacked_params, dim=0)

    global_model.load_state_dict(global_state)
    return global_model

# MAIN (optional standalone run)
def main():
    model_dir = "../data/models"

    # YOUR ACTUAL DIMENSIONS
    input_dim = 42      # From graph.num_node_features
    hidden_dim = 128    # Your trained models
    output_dim = 5      # 5 CKD stages

    model_A = GCN(input_dim, hidden_dim, output_dim).to(device)
    model_B = GCN(input_dim, hidden_dim, output_dim).to(device)
    model_C = GCN(input_dim, hidden_dim, output_dim).to(device)

    try:
        # YOUR ACTUAL LDP FILENAMES
        model_A.load_state_dict(torch.load(f"{model_dir}/model_A_ldp_eps0.5.pth", map_location=device))
        model_B.load_state_dict(torch.load(f"{model_dir}/model_B_ldp_eps2.5.pth", map_location=device))
        model_C.load_state_dict(torch.load(f"{model_dir}/model_C_ldp_eps1.5.pth", map_location=device))
        print(">>> Successfully received LDP hospital models.")
    except Exception as e:
        print(f"Error loading LDP models: {e}")
        return

    print(">>> Executing Privacy-Preserving FedAvg...")
    global_model = fedavg([model_A, model_B, model_C])

    save_path = os.path.join(model_dir, "global_model_ldp.pth")
    torch.save(global_model.state_dict(), save_path)
    print(f"Global LDP Model stored at: {save_path}")

if __name__ == "__main__":
    main()