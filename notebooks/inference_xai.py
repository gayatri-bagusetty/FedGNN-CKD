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
from notebooks.preprocessing import preprocess_single_patient

# 2. MODEL WRAPPER FOR EXPLAINER
class WrappedModel(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index):
        out = self.model(x, edge_index)
        return out.squeeze(0)   # ensure shape [2]

# 3. CLINICAL EXPLANATION GENERATOR
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
The model predicts Chronic Kidney Disease because the biomarker interaction
pattern resembles renal dysfunction.

• Serum Creatinine:
  High creatinine indicates reduced glomerular filtration rate (GFR).

• Albumin:
  Protein leakage into urine indicates glomerular damage.

• Specific Gravity:
  Abnormal urine concentration suggests tubular dysfunction.

• Hemoglobin:
  Low hemoglobin may indicate anemia associated with CKD.

• Sugar:
  Persistent glycosuria increases diabetic nephropathy risk.
"""

        recommendation = (
            "Evaluate eGFR, urine albumin-to-creatinine ratio, "
            "blood pressure control, and glycemic status."
        )

    else:

        clinical_explanation = """
The model predicts Non-CKD because renal biomarkers remain within
physiological ranges.

• Serum Creatinine indicates preserved kidney filtration.
• Albumin levels do not indicate proteinuria.
• Specific Gravity suggests normal urine concentration.
• Hemoglobin is not indicative of CKD-related anemia.
• Sugar levels do not suggest diabetic renal stress.
"""

        recommendation = (
            "Continue routine monitoring depending on patient risk factors."
        )

    return {
        "prediction": prediction,
        "confidence": f"{prob*100:.1f}%",
        "primary_biomarkers": ", ".join(feature_map.values()),
        "chart_values": percentages,
        "clinical_explanation": clinical_explanation,
        "recommendation": recommendation
    }

# 4. MAIN XAI INFERENCE FUNCTION
def run_inference_xai():
    feature_names = [
        'age','bp','sg','al','su','rbc','pc','pcc','ba',
        'bgr','bu','sc','sod','pot','hemo','pcv',
        'wbcc','rbcc','htn','dm','cad','appet','pe','ane'
    ]
    # Load Model
    model = GCN(input_dim=24, hidden_dim=32, output_dim=2)
    model_path = os.path.join(BASE_DIR,"data","models","global_model.pth")
    if not os.path.exists(model_path):
        print("global_model.pth not found")
        return
    model.load_state_dict(torch.load(model_path,map_location="cpu"))
    model.eval()
    print("Global model loaded.")

    # Example CKD Patient
    raw_patient = {
        'age':60,'bp':180,'sg':1.005,'al':4,'su':3,
        'rbc':'abnormal','pc':'abnormal','pcc':'present','ba':'present',
        'bgr':250,'bu':90,'sc':5.5,'sod':130,'pot':5.8,'hemo':8,
        'pcv':28,'wbcc':18000,'rbcc':3.0,'htn':'Yes','dm':'Yes',
        'cad':'Yes','appet':'poor','pe':'Yes','ane':'Yes'
    }
    # Preprocess Patient
    scaler_path = os.path.join(BASE_DIR,"data","processed","scaler.pkl")
    x_scaled = preprocess_single_patient(raw_patient)
    test_patient = torch.tensor(x_scaled,dtype=torch.float32).view(1,-1)

    # Graph structure (single node)
    edge_index = torch.empty((2,0),dtype=torch.long)

    # Prediction
    with torch.no_grad():
        logits = model(test_patient, edge_index)
        probs = F.softmax(logits.squeeze(0),dim=0)
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item()
    prediction = "CKD" if pred_class==1 else "Non-CKD"
    print("\nPrediction:",prediction)
    print("Confidence:",f"{confidence:.2%}")

    # Run GNNExplainer
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
    print("\nRunning GNNExplainer...")
    explanation = explainer(
        x=test_patient,
        edge_index=edge_index
    )
    # Feature Importance
    importances = explanation.node_mask.abs().squeeze().cpu().numpy()
    feat_imp = sorted(
        zip(feature_names,importances),
        key=lambda x:x[1],
        reverse=True
    )
    print("\nTop Biomarker Contributions:")
    for f,s in feat_imp[:5]:
        print(f"{f.upper():<10}: {s:.4f}")

    # Clinical Explanation
    report = generate_clinical_description(prediction,confidence,feat_imp)
    print("\nClinical Interpretation:")
    print(report["clinical_explanation"])
    print("\nRecommendation:")
    print(report["recommendation"])
    return report

# RUN SCRIPT
if __name__ == "__main__":
    run_inference_xai()