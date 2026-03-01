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
from notebooks.preprocessing import preprocess_single_patient

# --------------------------------------------------
# 2. MODEL WRAPPER FOR PYG EXPLAINER
# --------------------------------------------------
class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index):
        out = self.model(x, edge_index)
        return out[0]   # return logits for the single node


# --------------------------------------------------
# 3. CLINICAL EXPLANATION GENERATOR
# --------------------------------------------------
def generate_clinical_description(prediction, prob, feat_imp):

    feature_map = {
        "SG": "Specific Gravity",
        "AL": "Albumin",
        "SU": "Sugar",
        "SC": "Serum Creatinine",
        "HEMO": "Hemoglobin"
    }

    selected = []
    for name, value in feat_imp:
        key = name.upper()
        if key in feature_map:
            selected.append((key, abs(value)))

    final_values = []
    for key in feature_map.keys():
        found = next((v for k, v in selected if k == key), 0.0001)
        final_values.append(found)

    total = sum(final_values)
    percentages = [(v / total) * 100 for v in final_values]

    if prediction == "CKD":

        clinical_explanation = """
        The model predicts Chronic Kidney Disease because the combined biomarker
        interaction pattern resembles pathological renal dysfunction.

        • Serum Creatinine:
          Elevated creatinine suggests reduced glomerular filtration rate (GFR),
          indicating impaired kidney filtration.

        • Albumin:
          Presence of albumin in urine reflects glomerular membrane damage
          and protein leakage (proteinuria), a key marker of CKD.

        • Specific Gravity:
          Abnormal urine concentration indicates reduced tubular function
          and impaired renal concentrating ability.

        • Hemoglobin:
          Decreased hemoglobin may indicate anemia of chronic disease,
          commonly seen in chronic kidney impairment due to low erythropoietin.

        • Sugar:
          Persistent glycosuria suggests diabetic nephropathy risk,
          a major cause of CKD progression.
        """

        recommendation = (
            "Evaluate eGFR, urine ACR, blood pressure, "
            "and glycemic control. Consider nephrology referral and staging of CKD."
        )

    else:

        clinical_explanation = """
        The model predicts Non-CKD because renal-associated biomarkers
        appear within physiologically acceptable ranges and do not show
        pathological interaction patterns.

        • Serum Creatinine:
          Levels suggest preserved glomerular filtration function.

        • Albumin:
          Minimal or absent proteinuria indicates intact glomerular membrane integrity.

        • Specific Gravity:
          Normal urine concentration suggests preserved tubular function.

        • Hemoglobin:
          Hemoglobin levels do not indicate anemia related to renal insufficiency.

        • Sugar:
          No significant glycosuria-related renal stress detected.
        """

        recommendation = (
            "Continue routine monitoring based on "
            "patient risk factors such as hypertension or diabetes."
        )

    return {
        "prediction": prediction,
        "confidence": f"{prob*100:.1f}%",
        "primary_biomarkers": ", ".join(feature_map.values()),
        "chart_values": percentages,
        "clinical_explanation": clinical_explanation,
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

    model = GCN(input_dim=24, hidden_dim=32, output_dim=2)

    model_path = os.path.join(BASE_DIR, "data", "models", "global_model.pth")

    if not os.path.exists(model_path):
        print("global_model.pth not found.")
        return

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    ckd_patient = {
        'age': 60,
        'bp': 180,
        'sg': 1.005,
        'al': 4,
        'su': 3,
        'rbc': 'abnormal',
        'pc': 'abnormal',
        'pcc': 'present',
        'ba': 'present',
        'bgr': 250,
        'bu': 90,
        'sc': 5.5,
        'sod': 130,
        'pot': 5.8,
        'hemo': 8,
        'pcv': 28,
        'wbcc': 18000,
        'rbcc': 3.0,
        'htn': 'Yes',
        'dm': 'Yes',
        'cad': 'Yes',
        'appet': 'poor',
        'pe': 'Yes',
        'ane': 'Yes'
    }

    non_ckd_patient = {
        'age': 30,
        'bp': 120,
        'sg': 1.020,
        'al': 0,
        'su': 0,
        'rbc': 'normal',
        'pc': 'normal',
        'pcc': 'notpresent',
        'ba': 'notpresent',
        'bgr': 100,
        'bu': 20,
        'sc': 0.9,
        'sod': 140,
        'pot': 4.2,
        'hemo': 15,
        'pcv': 45,
        'wbcc': 8000,
        'rbcc': 5.0,
        'htn': 'No',
        'dm': 'No',
        'cad': 'No',
        'appet': 'good',
        'pe': 'No',
        'ane': 'No'
    }

    raw_patient = ckd_patient
    # raw_patient = non_ckd_patient

    scaler_path = os.path.join(BASE_DIR, "data", "processed", "scaler.pkl")
    x_scaled = preprocess_single_patient(raw_patient, scaler_path)

    test_patient = torch.tensor(x_scaled, dtype=torch.float32).view(1, -1)
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)

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
    # Prediction + confidence (FIXED)
    # --------------------------------------------------
    with torch.no_grad():
        logits = wrapped_model(test_patient, edge_index)   # shape: [2]
        probs = F.softmax(logits, dim=0)

        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item()

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

    report = generate_clinical_description(prediction, prob, feat_imp)

    print("\nPrediction:", prediction)
    print("Confidence:", f"{prob:.2%}")
    print("Explanation:", report)

    return report


if __name__ == "__main__":
    run_inference_xai()