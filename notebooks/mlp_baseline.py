import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================
# SIMPLE MLP MODEL
# ==========================
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=5):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# ==========================
# TRAIN FUNCTION
# ==========================
def train_mlp(X_train, y_train, X_val, y_val, epochs=50):

    model = MLP(X_train.shape[1]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Class imbalance handling
    class_counts = torch.bincount(torch.tensor(y_train)).float()
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            preds = val_outputs.argmax(dim=1).cpu()
            val_acc = accuracy_score(y_val.cpu(), preds)
            val_f1 = f1_score(y_val.cpu(), preds, average="macro")

        print(f"Epoch {epoch+1}/{epochs} | Val Acc: {val_acc:.4f} | Val MacroF1: {val_f1:.4f}")

    return model


# ==========================
# MAIN
# ==========================
def main():

    # Test only Hospital A first
    base_path = "../data/processed/hospital_A"

    train_df = pd.read_csv(os.path.join(base_path, "train.csv"))
    val_df = pd.read_csv(os.path.join(base_path, "val.csv"))

    X_train = train_df.drop("target", axis=1).values
    y_train = train_df["target"].values

    X_val = val_df.drop("target", axis=1).values
    y_val = val_df["target"].values

    print("Training MLP Baseline on Hospital A...")
    print("Train distribution:")
    print(train_df["target"].value_counts())

    print("\nVal distribution:")
    print(val_df["target"].value_counts())

    train_mlp(X_train, y_train, X_val, y_val)


if __name__ == "__main__":
    main()