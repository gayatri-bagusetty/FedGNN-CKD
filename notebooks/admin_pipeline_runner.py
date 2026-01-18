import os
import torch
import pandas as pd
from notebooks.preprocessing import preprocess_ckd_data
from notebooks.graph_construction import build_graph
from notebooks.local_gnn_training import train_with_global_weights
from notebooks.local_ldp import ldp_pipeline
from notebooks.fedavg_server import fedavg_pipeline
from notebooks.test_global_model import evaluate_global_model

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run_admin_flow(df: pd.DataFrame) -> float:
    """
    Runs full pipeline for dashboard CSV upload.

    Returns:
        float: noised accuracy immediately for dashboard
    """
    # 1️⃣ Preprocessing
    processed_df = preprocess_ckd_data(df)

    # 2️⃣ Graph Construction
    X = processed_df.drop("classification", axis=1)
    y = processed_df["classification"]
    new_graph = build_graph(X, y)

    # 3️⃣ Load global model if exists
    global_model_path = os.path.join(PROJECT_ROOT, "data/models/global_model.pth")
    global_model = torch.load(global_model_path, weights_only=False) if os.path.exists(global_model_path) else None

    # 4️⃣ Local Training
    local_model = train_with_global_weights(new_graph, model=global_model)

    # 5️⃣ Apply LDP
    eps_values = [3.0]
    ldp_results = ldp_pipeline(new_local_model=local_model, new_graph=new_graph, eps_values=eps_values)

    noised_model = ldp_results["new_graph"]["model"]
    noised_accuracy = ldp_results["new_graph"]["clean_accuracy"]

    # 6️⃣ Backend continuation (aggregates global model)
    backend_continuation(noised_model)

    return noised_accuracy


def backend_continuation(noised_model: torch.nn.Module):
    """
    Aggregates new noised model with old global model,
    updates the global model, and evaluates it.
    """
    global_model_path = os.path.join(PROJECT_ROOT, "data/models/global_model.pth")
    old_global_model = torch.load(global_model_path, weights_only=False) if os.path.exists(global_model_path) else None

    updated_global_model = fedavg_pipeline(new_noised_model=noised_model, old_global_model=old_global_model)
    acc, report = evaluate_global_model(updated_global_model)
    print(f"[INFO] Backend update complete - Global Model Accuracy: {acc:.4f}")