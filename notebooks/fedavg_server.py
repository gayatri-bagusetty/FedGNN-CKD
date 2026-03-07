import torch
import os
import copy
import sys
import random
import numpy as np

# Project imports
sys.path.append(os.path.abspath(".."))
from data.models.gcn_model import GCN

# REPRODUCIBILITY
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

set_seed(42)

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"FedAvg Server running on: {device}")

# TRUE WEIGHTED FEDAVG
def fedavg(models, data_sizes):
    """
    Weighted Federated Averaging
    models: list of local models
    data_sizes: number of samples per hospital
    """
    global_model = copy.deepcopy(models[0])
    global_state = global_model.state_dict()
    total_samples = sum(data_sizes)
    for key in global_state.keys():
        weighted_sum = 0
        for model, size in zip(models, data_sizes):
            weight = size / total_samples
            weighted_sum += weight * model.state_dict()[key].float()
        global_state[key] = weighted_sum
    global_model.load_state_dict(global_state)
    return global_model

# OPTIONAL INCREMENTAL UPDATE
def update_fedavg(local_model_path, global_model_path):
    input_dim = 24
    hidden_dim = 32
    output_dim = 2
    local_model = GCN(input_dim, hidden_dim, output_dim).to(device)
    local_model.load_state_dict(torch.load(local_model_path, map_location=device))
    global_model = GCN(input_dim, hidden_dim, output_dim).to(device)
    if os.path.exists(global_model_path):
        global_model.load_state_dict(torch.load(global_model_path, map_location=device))
    alpha = 0.8
    beta = 0.2
    global_state = global_model.state_dict()
    local_state = local_model.state_dict()
    for key in global_state:
        global_state[key] = alpha * global_state[key] + beta * local_state[key]
    global_model.load_state_dict(global_state)
    torch.save(global_model.state_dict(), global_model_path)
    print("Incremental FedAvg update complete")

# MAIN FEDERATED ROUND
def main():
    model_dir = "../data/models"
    input_dim = 24
    hidden_dim = 32
    output_dim = 2
    # Load hospital models
    model_A = GCN(input_dim, hidden_dim, output_dim).to(device)
    model_B = GCN(input_dim, hidden_dim, output_dim).to(device)
    model_C = GCN(input_dim, hidden_dim, output_dim).to(device)
    try:
        model_A.load_state_dict(torch.load(f"{model_dir}/model_A_ldp.pth", map_location=device))
        model_B.load_state_dict(torch.load(f"{model_dir}/model_B_ldp.pth", map_location=device))
        model_C.load_state_dict(torch.load(f"{model_dir}/model_C_ldp.pth", map_location=device))
        print("Hospital models received")
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    # Example dataset sizes (change if needed)
    data_sizes = [280, 280, 210]
    print("Executing Weighted FedAvg...")
    global_model = fedavg(
        [model_A, model_B, model_C],
        data_sizes
    )
    save_path = os.path.join(model_dir, "global_model.pth")
    torch.save(global_model.state_dict(), save_path)
    print(f"Global model saved → {save_path}")

if __name__ == "__main__":
    main()