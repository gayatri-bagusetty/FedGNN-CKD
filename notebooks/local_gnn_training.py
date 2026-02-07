import torch
import torch.nn.functional as F
import os
import sys
from sklearn.metrics import accuracy_score
from notebooks.resource_monitor import ResourceMonitor


# Ensure pathing for GCN model import
sys.path.append(os.path.abspath(".."))
from data.models.gcn_model import GCN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_local_model(graph_path, epochs=100, lr=0.01, hospital_name="Hospital"):
    if not os.path.exists(graph_path):
        print(f"Error: Graph file {graph_path} not found.")
        return None

    graph = torch.load(graph_path, weights_only=False).to(device)

    model = GCN(
        input_dim=graph.num_node_features,
        hidden_dim=32,
        output_dim=2
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # RESOURCE MONITOR
    monitor = ResourceMonitor()
    monitor.start_timer()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(graph.x, graph.edge_index)
        loss = criterion(out, graph.y)
        loss.backward()
        optimizer.step()

    training_time = monitor.stop_timer()
    memory_used = monitor.memory_usage_mb()

    print(f"\n[{hospital_name}] RESOURCE USAGE")
    print(f"Training Time   : {training_time:.2f} seconds")
    print(f"Memory Usage    : {memory_used:.2f} MB")

    return model

def evaluate_on_val(model, hospital_id):
    """Evaluates the trained model on the validation CSV for that hospital."""
    val_path = f"../data/processed/hospital_{hospital_id}/val.csv"
    if not os.path.exists(val_path):
        return 0.0
    
    df_val = pd.read_csv(val_path)
    X_val = torch.tensor(df_val.drop('classification', axis=1).values, dtype=torch.float).to(device)
    y_val = torch.tensor(df_val['classification'].values, dtype=torch.long).to(device)
    
    # Validation uses a 'dummy' edge index for node-level inference if graph not built
    # In GCN, we need edges, but for simple val we can use self-loops or empty
    edge_index = torch.zeros((2, 0), dtype=torch.long).to(device) 

    model.eval()
    with torch.no_grad():
        logits = model(X_val, edge_index)
        preds = logits.argmax(dim=1).cpu()
        labels = y_val.cpu()

    return accuracy_score(labels, preds)

def main():
    graph_dir = "../data/graph"
    model_save_dir = "../data/models"
    os.makedirs(model_save_dir, exist_ok=True)

    hospitals = ['A', 'B', 'C']

    for h in hospitals:
        graph_path = os.path.join(graph_dir, f"graph_{h}.pt")
        print(f"\n--- Training Hospital {h} ---")

        model = train_local_model(
            graph_path,
            hospital_name=f"Hospital {h}"
        )

        if model:
            save_path = os.path.join(model_save_dir, f"model_{h}.pth")
            torch.save(model.state_dict(), save_path)

            # COMMUNICATION COST
            from resource_monitor import ResourceMonitor
            size_mb = ResourceMonitor.model_size_mb(save_path)
            print(f"Model Communication Cost: {size_mb:.2f} MB")


    print("\nLocal models saved successfully.")

if __name__ == "__main__":
    import pandas as pd # Required for evaluation function
    main()