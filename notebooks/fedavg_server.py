import torch
import os
import copy
import sys

# Add project path to access the model architecture
sys.path.append(os.path.abspath("../data"))
from models.gcn_model import GCN

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Server acting on device: {device}")

def fedavg(ldp_models):
    """
    Aggregates LDP-protected local models into a global model update
    using Federated Averaging (FedAvg).

    Args:
        ldp_models (list): List of LDP-trained PyTorch models.

    Returns:
        global_model (torch.nn.Module): The newly aggregated global model.
    """
    # Initialize global model with the same architecture as the first local model
    global_model = copy.deepcopy(ldp_models[0])
    global_state = global_model.state_dict()

    # Iterate through each parameter (weights and biases)
    for key in global_state.keys():
        # Stack the same parameter from all models and calculate the mean
        # Note: .float() ensures precision during averaging
        stacked_params = torch.stack([model.state_dict()[key].float() for model in ldp_models])
        global_state[key] = torch.mean(stacked_params, dim=0)

    # Load the averaged parameters back into the global model structure
    global_model.load_state_dict(global_state)
    return global_model

def main():
    # 1. Setup Paths
    model_dir = "../data/models"
    graph_sample_path = "../data/graph/graph_A.pt"

    if not os.path.exists(graph_sample_path):
        print("Error: Sample graph not found. Cannot determine input dimensions.")
        return

    # 2. Initialize architecture
    # We load a sample graph strictly to define the input_dim (features)
    sample_graph = torch.load(graph_sample_path, weights_only=False)
    input_dim = sample_graph.num_node_features
    
    # Initialize empty model instances
    model_A = GCN(input_dim, 32, 2).to(device)
    model_B = GCN(input_dim, 32, 2).to(device)
    model_C = GCN(input_dim, 32, 2).to(device)

    # 3. Load the LDP-Protected weights sent by hospitals
    try:
        model_A.load_state_dict(torch.load(f"{model_dir}/model_A_ldp.pth"))
        model_B.load_state_dict(torch.load(f"{model_dir}/model_B_ldp.pth"))
        model_C.load_state_dict(torch.load(f"{model_dir}/model_C_ldp.pth"))
        print("Successfully loaded LDP-protected local models.")
    except FileNotFoundError as e:
        print(f"Error: Missing local model files. {e}")
        return

    # 4. Perform Federated Averaging
    print("Aggregating models via FedAvg...")
    global_model = fedavg([model_A, model_B, model_C])

    # 5. Save Global Model
    save_path = f"{model_dir}/global_model.pth"
    torch.save(global_model.state_dict(), save_path)

    print(f"Global model update created and stored at: {save_path}")

if __name__ == "__main__":
    main()