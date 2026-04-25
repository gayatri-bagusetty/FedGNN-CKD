import os
import sys
import time
import shutil
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

# Path setup
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from Client.preprocessing import preprocess_uploaded_dataset
from Client.graph_construction import construct_graph
from Client.local_gnn_training import train_local_model
from Client.local_ldp import select_best_epsilon
from data.models.gcn_model import GCN


def federated_average(global_model, local_model):
    """Perform FedAvg aggregation"""
    global_weights = global_model.state_dict()
    local_weights = local_model.state_dict()

    for key in global_weights:
        global_weights[key] = (global_weights[key] + local_weights[key]) / 2

    global_model.load_state_dict(global_weights)
    return global_model

def split_dataset(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"] if "label" in df.columns else None
    )
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    return train_path, val_path

def run_admin_fl_update(csv_path):
    logs = []
    start_time = time.time()
    # ADMIN DIRECTORY STRUCTURE
    admin_base = os.path.join(ROOT_DIR, "data", "admin")
    upload_dir = os.path.join(admin_base, "uploads")
    graph_dir = os.path.join(admin_base, "graphs")
    model_dir = os.path.join(admin_base, "models")

    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # STEP 1 : STORE UPLOADED CSV
    filename = os.path.basename(csv_path)
    stored_csv = os.path.join(upload_dir, filename)
    shutil.copy(csv_path, stored_csv)
    logs.append(f"Uploaded CSV stored at: {stored_csv}")

    # STEP 2 : PREPROCESS DATA
    logs.append("Preprocessing uploaded dataset...")
    preprocessed_csv_path = preprocess_uploaded_dataset(stored_csv)
    logs.append(f"Dataset processed and saved to: {preprocessed_csv_path}")
    
    # STEP 2  Split dataset
    logs.append("Splitting dataset...")
    train_csv, val_csv = split_dataset(preprocessed_csv_path, upload_dir)

    # STEP 3 : BUILD GRAPHS
    logs.append("Constructing graphs...")
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    k_train = min(10, len(train_df) - 1)
    k_val = min(10, len(val_df) - 1)
    train_graph = construct_graph(train_csv, k_train)
    val_graph = construct_graph(val_csv, k_val)
    train_graph_path = os.path.join(graph_dir, "admin_train_graph.pt")
    val_graph_path = os.path.join(graph_dir, "admin_val_graph.pt")

    torch.save(train_graph, train_graph_path)
    torch.save(val_graph, val_graph_path)
    logs.append(f"Train graph saved: {train_graph_path}")
    logs.append(f"Validation graph saved: {val_graph_path}")

    # STEP 4 : LOCAL GNN TRAINING
    logs.append("Training local GNN model...")
    model, train_acc_list, val_acc_list, training_time, memory_used, val_acc, precision, recall, f1= train_local_model(
        train_graph_path,
        val_graph_path,
        hospital_name="Admin Hospital"
    )
    clean_accuracy = val_acc
    logs.append(f"Clean accuracy before privacy: {clean_accuracy:.4f}")

    # STEP 5 : APPLY LOCAL DIFFERENTIAL PRIVACY
    logs.append("Applying Local Differential Privacy...")
    train_graph = torch.load(train_graph_path, weights_only=False)
    eps_values = [0.5, 1, 1.5, 2, 2.5, 3]
    ldp_model, best_eps, train_acc, clean_acc = select_best_epsilon(
        model,
        train_graph,
        eps_values,
        hospital_name="Admin Hospital"
    )
    logs.append(f"LDP applied with optimal ε = {best_eps}")

    # STEP 6 : LOAD GLOBAL MODEL
    global_model = GCN(
        input_dim=24,
        hidden_dim=32,
        output_dim=2
    )
    global_model_path = os.path.join(ROOT_DIR, "Server", "models", "global_model.pth")
    if not os.path.exists(global_model_path):
        raise FileNotFoundError("Global model not found!")
    global_weights = torch.load(global_model_path)
    global_model.load_state_dict(global_weights)
    logs.append("Loaded existing global model")

    # STEP 7 : FEDERATED AGGREGATION
    logs.append("Performing Federated Averaging...")
    updated_global_model = federated_average(global_model, ldp_model)
    torch.save(updated_global_model.state_dict(), global_model_path)
    logs.append("Global model updated successfully")

    # STEP 8 : SAVE ADMIN LOCAL MODEL
    local_model_path = os.path.join(model_dir, "admin_local_model.pt")
    torch.save(ldp_model.state_dict(), local_model_path)
    logs.append(f"Admin local model stored at: {local_model_path}")

    # FINAL
    total_time = time.time() - start_time
    logs.append(f"Total pipeline time: {total_time:.2f} sec")
    return clean_accuracy, logs