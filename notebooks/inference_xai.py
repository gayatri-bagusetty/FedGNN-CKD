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
        probs = torch.softmax(out, dim=1)
        return probs[:, 1]  # CKD probability


# --------------------------------------------------
# 3. CLINICAL EXPLANATION GENERATOR
# --------------------------------------------------
def generate_clinical_description(prediction, prob, feat_imp):

    filtered_features = [f for f in feat_imp if f[0].upper() != "AGE"]
    top_features = filtered_features[:4]
    top_names = [f[0].upper() for f in top_features]

    confidence = f"{prob*100:.1f}%"

    if prediction == "CKD":
        explanation_text = (
            f"Clinical Decision Support Summary:\n\n"
            f"The model indicates a HIGH likelihood of Chronic Kidney Disease "
            f"(confidence: {confidence}).\n"
            f"Key contributing clinical parameters include:\n"
            f"- {top_names[0]} (primary driver)\n"
            f"- {top_names[1] if len(top_names)>1 else ''}\n"
            f"- {top_names[2] if len(top_names)>2 else ''}\n"
            f"These parameters are commonly associated with impaired renal function, "
            f"electrolyte imbalance, or abnormal filtration markers. "
            f"Correlation patterns resemble profiles observed in CKD cohorts."
        )

        recommendation = (
            "Suggested clinical consideration: Correlate with serum creatinine, "
            "eGFR, and urine analysis findings. Consider nephrology referral if indicated."
        )

    else:
        explanation_text = (
            f"Clinical Decision Support Summary:\n"
            f"The model indicates LOW likelihood of Chronic Kidney Disease "
            f"(confidence: {confidence}).\n\n"
            f"Renal-associated parameters appear within acceptable clinical range. "
            f"No strong pathological interaction patterns detected.\n"
            f"Primary monitored biomarkers:\n"
            f"- {top_names[0]}\n"
            f"- {top_names[1] if len(top_names)>1 else ''}\n"
            f"- {top_names[2] if len(top_names)>2 else ''}"
        )

        recommendation = (
            "Suggested clinical consideration: Continue routine monitoring "
            "based on patient risk profile."
        )

    return {
        "prediction": prediction,
        "confidence": confidence,
        "primary_biomarkers": ", ".join(top_names),
        "clinical_explanation": explanation_text,
        "recommendation": recommendation
    }


# --------------------------------------------------
# 4. STANDALONE TEST INFERENCE
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

    model_path = os.path.join(BASE_DIR, "data", "models", "global_model.pth")

    if not os.path.exists(model_path):
        print("global_model.pth not found.")
        return

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # --------------------------------------------------
    # Example patient vector (already scaled)
    # --------------------------------------------------
    ckd_patient = torch.tensor(
        [[
            0.70, 0.75, 1.005, 0.8, 0.6, 0, 0, 1, 1,
            0.80, 0.85, 0.90, 0.30, 0.80, 0.25, 0.20,
            0.85, 0.30, 1, 1, 1, 0, 1, 1
        ]],
        dtype=torch.float32
    )
    
    non_ckd_patient = torch.tensor(
        [[
            0.30, 0.35, 1.020, 0.0, 0.0, 1, 1, 0, 0,
            0.25, 0.20, 0.10, 0.65, 0.35, 0.85, 0.90,
            0.20, 0.85, 0, 0, 0, 1, 0, 0
        ]],
        dtype=torch.float32
    )
    
    test_patient = ckd_patient
    # test_patient = non_ckd_patient

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
    # Prediction + confidence
    # --------------------------------------------------
    with torch.no_grad():
        logits = model(test_patient, edge_index)
        probs = torch.softmax(logits, dim=1)
        confidence = probs.max(dim=1)[0].item()
        pred_class = probs.argmax(dim=1).item()

    prediction = "CKD" if pred_class == 1 else "Non-CKD"
    prob = confidence

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
    report = generate_clinical_description(prediction, prob, feat_imp)

    print("\nPrediction:", prediction)
    print("Confidence:", f"{prob:.2%}")
    print("Explanation:", report)
    return report


if __name__ == "__main__":
    run_inference_xai()