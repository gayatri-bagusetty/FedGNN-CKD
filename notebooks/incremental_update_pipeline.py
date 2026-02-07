import os
import sys
import torch
from sklearn.metrics import accuracy_score
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from notebooks.preprocessing import preprocess_uploaded_dataset
from notebooks.graph_construction import build_graph
from notebooks.local_gnn_training import train_local_model
from notebooks.local_ldp import apply_ldp
from notebooks.fedavg_server import update_fedavg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_incremental_update(csv_path):
    """
    Workflow:
    CSV → Preprocess → Graph → Train → LDP → FedAvg → Return accuracy + status text
    """

    status_log = []

    # 1. PREPROCESS
    status_log.append("🟢 Preprocessing uploaded CSV...")
    X_scaled, y = preprocess_uploaded_dataset(csv_path)
    status_log.append("✅ Preprocessing completed.")

    X_scaled = np.asarray(X_scaled, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    X_scaled = np.nan_to_num(X_scaled)

    # 2. GRAPH CONSTRUCTION
    status_log.append("🟡 Building graph from processed data...")
    graph = build_graph(X_scaled, y, k=5)
    status_log.append("✅ Graph construction completed.")

    temp_graph_path = os.path.join(ROOT_DIR, "data/graph/temp_uploaded_graph.pt")
    os.makedirs(os.path.dirname(temp_graph_path), exist_ok=True)
    torch.save(graph, temp_graph_path)

    # 3. LOCAL TRAINING
    status_log.append("🟠 Training local model using global model weights...")
    local_model = train_local_model(
        graph_path=temp_graph_path,
        hospital_name="New Hospital"
    )
    status_log.append("✅ Local training completed.")

    # 4. ACCURACY
    local_model.eval()
    with torch.no_grad():
        out = local_model(graph.x.to(device), graph.edge_index.to(device))
        preds = out.argmax(dim=1).cpu()
        labels = graph.y.cpu()
        accuracy = accuracy_score(labels, preds)

    status_log.append(f"📊 Local Training Accuracy: {accuracy:.4f}")

    # 5. APPLY LDP
    status_log.append("🔐 Applying Local Differential Privacy (ε = 3.0)..")
    ldp_model = apply_ldp(local_model, epsilon=3.0)

    ldp_model_path = os.path.join(ROOT_DIR, "data/models/new_local_ldp.pth")
    os.makedirs(os.path.dirname(ldp_model_path), exist_ok=True)
    torch.save(ldp_model.state_dict(), ldp_model_path)
    status_log.append("✅ LDP applied to local model.")

    # 6. FED SERVER AGGREGATION
    print(">>> Sending to Fed Server...")
    global_model_path = os.path.join(ROOT_DIR, "data/models/global_model.pth")
    update_fedavg(ldp_model_path, global_model_path)
    print("✅ Global model updated")

    status_log.append("✅ Global model updated using FedAvg.")

    return accuracy, status_log