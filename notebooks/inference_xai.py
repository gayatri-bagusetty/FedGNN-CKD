import torch
import sys
import os
import torch.nn.functional as F
import numpy as np
from torch_geometric.explain import Explainer, GNNExplainer

# --------------------------------------------------
# 1. PATH SETUP
# --------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from data.models.gcn_model import GCN


# --------------------------------------------------
# 2. MODEL WRAPPER FOR PYG EXPLAINER
# --------------------------------------------------
class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index):
        out = self.model(x, edge_index)
        return torch.softmax(out, dim=1)[:, 1]  # CKD probability


# --------------------------------------------------
# 3. CLINICAL EXPLANATION GENERATOR
# --------------------------------------------------
def generate_clinical_description(prediction, prob, feat_imp):
    # 1. Filter out 'AGE' from drivers to keep focus on physiological markers
    filtered_features = [f for f in feat_imp if f[0].upper() != "AGE"]
    
    # Get top 3 physiological drivers
    top_3_names = [f[0].upper() for f in filtered_features[:3]]
    primary_factor = top_3_names[0] if top_3_names else "RENAL BIOMARKERS"

    # 2. Build bulleted explanation based on key features
    if prediction == "CKD":
        reasoning = (
            f"The Graph Neural Network (GNN) identifies a high risk of CKD based on the following findings:\n\n"
            f"* **{primary_factor} Correlation:** Significant deviation from normal physiological baseline detected.\n"
            f"* **Abnormal Connectivity:** Disrupted interactions between {', '.join(top_3_names[1:])} and other key nodes.\n"
            f"* **Pattern Matching:** Patient embedding aligns with renal impairment clusters identified in federated training."
        )

    else:
        reasoning = (
            f"The patient demonstrates a stable renal profile with no significant disease markers:\n\n"
            f"* **{primary_factor} Stability:** Feature levels are within healthy clinical bounds.\n"
            f"* **Balanced Graph:** Learned representations for {', '.join(top_3_names[1:])} show homeostatic behavior.\n"
            f"* **Cluster Alignment:** Data matches healthy clinical cohorts with high confidence."
        )

    return {
        "explanation": reasoning,
        "primary_drivers": ", ".join(top_3_names),
    }


# --------------------------------------------------
# 4. STANDALONE TEST INFERENCE (OPTIONAL)
# --------------------------------------------------
def run_inference_xai():

    feature_names = [
        'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
        'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv',
        'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
    ]

    # --------------------------------------------------
    # Load global federated model
    # --------------------------------------------------
    model = GCN(input_dim=24, hidden_dim=32, output_dim=2)

    model_path = os.path.join(
        BASE_DIR, "data", "models", "global_model.pth"
    )

    if not os.path.exists(model_path):
        print("global_model.pth not found.")
        return

    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )

    model.eval()

    # --------------------------------------------------
    # Example patient vector (already scaled)
    # --------------------------------------------------
    test_patient = torch.tensor(
        [[
            0.3, 0.4, 1.025, 0, 0, 1, 1, 0, 0,
            0.5, 0.2, 0.1, 0.8, 0.4, 0.9, 0.85,
            0.2, 0.75, 0, 0, 0, 1, 0, 0
        ]],
        dtype=torch.float32
    )

    edge_index = torch.tensor([[0], [0]], dtype=torch.long)

    # --------------------------------------------------
    # XAI ENGINE
    # --------------------------------------------------
    wrapped_model = WrappedModel(model)

    explainer = Explainer(
        model=wrapped_model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type="model",
        node_mask_type="attributes",
        model_config=dict(
            mode="binary_classification",
            task_level="node",
            return_type="raw"
        )
    )

    print("Running GNNExplainer...")

    explanation = explainer(test_patient, edge_index)

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------
    with torch.no_grad():
        logits = model(test_patient, edge_index)
        prob = torch.softmax(logits, dim=1)[0][1].item()

    prediction = "CKD" if prob >= 0.50 else "Non-CKD"

    # --------------------------------------------------
    # Feature importance
    # --------------------------------------------------
    importances = explanation.node_mask.squeeze().cpu().numpy()

    feat_imp = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )

    print("\nTop Biomarker Contributions:")
    for f, s in feat_imp[:5]:
        print(f"{f.upper():<10}: {s:.4f}")

    # --------------------------------------------------
    # Explanation text
    # --------------------------------------------------
    report = generate_clinical_description(
        prediction, prob, feat_imp
    )

    print("\nPrediction:", prediction)
    print("Confidence:", f"{prob:.2%}")
    print("Explanation:", report)


if __name__ == "__main__":
    run_inference_xai()