import torch
import os
import sys
import copy

# -------------------------------------------------
# PATH
# -------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../data"))
from models.gcn_model import GCN

# -------------------------------------------------
# DEVICE
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------------------------------------
# FEDAVG CORE
# -------------------------------------------------
def fedavg(models):
    """
    Standard FedAvg aggregation.
    models: list of PyTorch models
    """
    global_model = copy.deepcopy(models[0])
    global_dict = global_model.state_dict()

    for key in global_dict.keys():
        global_dict[key] = torch.stack(
            [m.state_dict()[key].float() for m in models],
            dim=0
        ).mean(dim=0)

    global_model.load_state_dict(global_dict)
    return global_model


# -------------------------------------------------
# MAIN ADMIN PIPELINE
# -------------------------------------------------
def fedavg_pipeline(
        new_noised_model=None,
        old_global_model=None
):
    """
    ADMIN PIPELINE PATTERN

    Case 1:
        new_noised_model provided
        → aggregate with old global model
        → replace global model

    Case 2:
        no input
        → aggregate hospital LDP models
    """

    model_dir = "../data/models"
    os.makedirs(model_dir, exist_ok=True)

    # =====================================================
    # CASE 1 — CONTINUAL UPDATE MODE
    # =====================================================
    if new_noised_model is not None and old_global_model is not None:

        print("\n--- Aggregating new update with global model ---")

        aggregated_model = fedavg(
            [old_global_model, new_noised_model]
        )

        torch.save(
            aggregated_model.state_dict(),
            os.path.join(model_dir, "global_model.pth")
        )

        print("Global model updated and replaced.")

        return aggregated_model


    # =====================================================
    # CASE 2 — INITIAL FEDERATED ROUND
    # =====================================================
    print("\n--- Aggregating hospital LDP models ---")

    hospital_models = []

    for h in ["A", "B", "C"]:
        path = os.path.join(model_dir, f"model_{h}_ldp.pth")

        if not os.path.exists(path):
            print(f"Missing {path}")
            continue

        model = torch.load(path, weights_only=False)
        hospital_models.append(model)

    if len(hospital_models) == 0:
        raise RuntimeError("No hospital models found for FedAvg")

    global_model = fedavg(hospital_models)

    torch.save(
        global_model.state_dict(),
        os.path.join(model_dir, "global_model.pth")
    )

    print("Initial global model created.")

    return global_model


# -------------------------------------------------
# TEST
# -------------------------------------------------
if __name__ == "__main__":
    global_model = fedavg_pipeline()
    print("FedAvg completed.")