import os
import sys
import time
import torch

# -------------------------------------------------------
# Path setup
# -------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from notebooks.preprocessing import preprocess_uploaded_dataset
from notebooks.graph_construction import construct_graph
from notebooks.local_gnn_training import train_local_model
from notebooks.local_ldp import select_best_epsilon

# -------------------------------------------------------
# Main Pipeline
# -------------------------------------------------------
def run_incremental_update(csv_path):

    logs = []

    start = time.time()

    logs.append("Preprocessing uploaded CSV...")
    processed_path = preprocess_uploaded_dataset(csv_path)
    logs.append("Preprocessing completed.")

    logs.append("Building graph from processed data...")
    train_graph_path, val_graph_path = construct_graph(processed_path)
    logs.append("Graph construction completed.")

    logs.append("Training local model...")
    model, train_acc_list, val_acc_list = train_local_model(
        train_graph_path,
        val_graph_path,
        hospital_name="Hospital A"
    )
    logs.append("Local training completed.")

    # ================================
    # CLEAN ACCURACY (BEFORE LDP)
    # ================================
    clean_accuracy = val_acc_list[-1]
    logs.append(f"Clean model accuracy (before LDP): {clean_accuracy:.4f}")

    # ================================
    # APPLY LDP
    # ================================
    logs.append("Applying Local Differential Privacy (ε = 3.0)...")

    train_graph = torch.load(train_graph_path, weights_only=False)

    eps_values = [0.5, 1, 1.5, 2, 2.5, 3]

    best_ldp_model, best_eps, accs, clean_acc = select_best_epsilon(
        model,
        train_graph,
        eps_values,
        hospital_name="Hospital A"
    )

    logs.append(f"LDP applied with optimal ε = {best_eps}")

    end = time.time()
    logs.append(f"Total pipeline time: {end - start:.2f} sec")

    # ================================
    # RETURN CLEAN ACCURACY
    # ================================
    return clean_accuracy, logs