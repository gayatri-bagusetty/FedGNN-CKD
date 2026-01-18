import torch
import torch.nn.functional as F
import os
import sys
from sklearn.metrics import accuracy_score

# Ensure the models directory is in the path for importing GCN
sys.path.append(os.path.abspath("../data"))
from models.gcn_model import GCN

# Check Device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train_local_model(graph_path, epochs=100, lr=0.01):
    """
    Loads a hospital graph and trains a local GCN model.
    """
    if not os.path.exists(graph_path):
        print(f"Error: Graph file {graph_path} not found.")
        return None

    # Load graph to device
    graph = torch.load(graph_path, weights_only=False).to(device)

    # Initialize Model
    model = GCN(
        input_dim=graph.num_node_features,
        hidden_dim=32,
        output_dim=2
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Training Loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(graph.x, graph.edge_index)
        loss = criterion(out, graph.y)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"[{os.path.basename(graph_path)}] Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    return model

def evaluate_model(model, graph_path):
    """
    Evaluates the performance of a model on a given graph.
    """
    if model is None or not os.path.exists(graph_path):
        return 0.0

    graph = torch.load(graph_path, weights_only=False).to(device)
    model.eval()

    with torch.no_grad():
        logits = model(graph.x, graph.edge_index)
        preds = logits.argmax(dim=1).cpu()
        labels = graph.y.cpu()

    return accuracy_score(labels, preds)

def main():
    # 1. Define paths
    graph_dir = "../data/graph"
    model_save_dir = "../data/models"
    os.makedirs(model_save_dir, exist_ok=True)

    hospitals = ['A', 'B', 'C']
    trained_models = {}

    # 2. Train and Evaluate for each hospital
    for h in hospitals:
        graph_path = os.path.join(graph_dir, f"graph_{h}.pt")
        print(f"\n--- Starting Training for Hospital {h} ---")
        
        model = train_local_model(graph_path)
        
        if model:
            acc = evaluate_model(model, graph_path)
            print(f"Hospital {h} Accuracy: {acc:.4f}")
            
            # 3. Save weights (State Dict)
            save_path = os.path.join(model_save_dir, f"model_{h}.pth")
            torch.save(model.state_dict(), save_path)
            trained_models[h] = model

    print("\nLocal models saved successfully in ../data/models/")

if __name__ == "__main__":
    main()