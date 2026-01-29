import torch
import sys
import os
import torch.nn.functional as F
import numpy as np
from torch_geometric.explain import Explainer, GNNExplainer

# 1. PATH SETUP
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)
from data.models.gcn_model import GCN

# 2. MODEL WRAPPER FOR PYG EXPLAINER
class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x, edge_index):
        out = self.model(x, edge_index)
        return torch.softmax(out, dim=1)[:, 1] # Target CKD class probability

# --- Updated function in inference_xai.py ---

def generate_clinical_description(prediction, prob, feat_imp):
    top_3 = [f[0].upper() for f in feat_imp[:3]]
    is_positive = prob >= 0.50
    # Reasoning logic - Clean text for web display
    if is_positive:
        reasoning = (
            "Significant biomarker deviations detected. The GNN model identifies patterns "
            "highly consistent with Chronic Kidney Disease clusters. Clinical correlation is advised."
        )
    else:
        reasoning = (
            "Biomarker levels are within stable physiological ranges. The patient profile "
            "matches healthy clinical clusters within the latent space."
        )

    # Return a dictionary. This allows dashboard_doctor.py to access specific keys.
    return {
        "explanation": reasoning,
        "primary_drivers": ", ".join(top_3)
    }

def run_inference_xai():
    # 4. FULL BIOMARKER LIST (24 Features)
    feature_names = [
        'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu',
        'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad',
        'appet', 'pe', 'ane'
    ]
    
    # 5. LOAD GLOBAL MODEL
    model = GCN(input_dim=24, hidden_dim=32, output_dim=2)
    model_path = os.path.join(BASE_DIR, 'data', 'models', 'global_model.pth')
    
    if not os.path.exists(model_path):
        print("Error: global_model.pth not found. Complete the FedAvg step first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # 6. TEST DATA (Scaled 24-feature vector)
    healthy_test = [
        0.3, 0.4, 1.025, 0, 0, 1, 1, 0, 0, 0.5, 0.2, 0.1, 0.8, 0.4, 0.9, 0.85, 0.2, 0.75, 0, 0, 0, 1, 0, 0
    ]
    
    test_patient = torch.tensor([healthy_test], dtype=torch.float)
    edge_index = torch.tensor([[0], [0]], dtype=torch.long) 

    # 7. XAI ENGINE INITIALIZATION
    wrapped_model = WrappedModel(model)
    explainer = Explainer(
        model=wrapped_model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        model_config=dict(
            mode='binary_classification', # Updated to binary
            task_level='node', 
            return_type='raw'
        ),
    )

    print(">>> GNNExplainer is analyzing the clinical biomarkers...")
    explanation = explainer(test_patient, edge_index)
    
    # 8. UPDATED PREDICTION LOGIC
    with torch.no_grad():
        logits = model(test_patient, edge_index)
        prob_ckd = torch.softmax(logits, dim=1)[0][1].item()
        
        # Binary result thresholds
        if prob_ckd >= 0.50:
            prediction = "High Risk — CKD"
        else:
            prediction = "Low Risk — Non-CKD"

    # 9. FEATURE IMPORTANCE & REPORT
    importances = explanation.node_mask.squeeze().cpu().numpy()
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    print("\nTOP BIOMARKER INFLUENCE:")
    for name, imp in feat_imp[:5]:
        print(f"{name.upper():<10}: {imp:.4f}")
    
    # Pass to binary reasoning engine
    print("\nGENERATING UI REPORT...")
    print(generate_clinical_description(prediction, prob_ckd, feat_imp))

if __name__ == "__main__":
    run_inference_xai()