import torch
import os
import sys
import numpy as np
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import ModelMode

model_config=dict(
    mode=ModelMode.binary_classification,
    task_level="node",
    return_type="raw"
)


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))
sys.path.append(os.path.join(BASE_DIR, "notebooks"))

from data.models.gcn_model import GCN
from preprocessing import preprocess_single_patient
from graph_construction import build_single_node_graph
from inference_xai import generate_clinical_description


class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index):
        out = self.model(x, edge_index)
        return torch.softmax(out, dim=1)[:,1]


class ClinicalInferenceEngine:
    def __init__(self, model_path, scaler_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler_path = scaler_path

        self.feature_names = [
            'age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu',
            'sc','sod','pot','hemo','pcv','wbcc','rbcc','htn','dm','cad',
            'appet','pe','ane'
        ]

        self.model = GCN(input_dim=24, hidden_dim=32, output_dim=2).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.wrapped_model = WrappedModel(self.model)

        self.explainer = Explainer(
            model=self.wrapped_model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type="model",
            node_mask_type="attributes",
            model_config=dict(
                mode=ModelMode.binary_classification,
                task_level="node",
                return_type="raw"
            )
        )

    def run_diagnosis(self, raw_data_dict):

        # 1. Preprocessing
        x_scaled = preprocess_single_patient(raw_data_dict, self.scaler_path)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(self.device)

        # 2. Graph
        x, edge_index = build_single_node_graph(x_tensor)
        x, edge_index = x.to(self.device), edge_index.to(self.device)

        # 3. Inference
        with torch.no_grad():
            logits = self.model(x, edge_index)
            prob = torch.softmax(logits, dim=1)[0][1].item()
            if prob < 0.30:
                prediction = "Low Risk — Non-CKD"
            elif prob < 0.50:
                prediction = "Mild Risk — Non-CKD"
            elif prob < 0.70:
                prediction = "Moderate Risk — CKD"
            else:
                prediction = "High Risk — CKD"

        # 4. XAI
        explanation = self.explainer(x, edge_index)
        importances = explanation.node_mask.squeeze().cpu().numpy()

        feat_imp = sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )

        report = generate_clinical_description(prediction, prob, feat_imp)

        return {
            "prediction": prediction,
            "probability": prob,
            "top_features": feat_imp[:5],
            "report": report
        }

if __name__ == "__main__":
    model_path = os.path.join("data", "models", "global_model.pth")
    scaler_path = os.path.join("data", "processed", "scaler.pkl")

    engine = ClinicalInferenceEngine(model_path, scaler_path)

    test_patient = {
        'age': 45, 'bp': 80, 'sg': 1.020, 'al': 1, 'su': 0, 'rbc': 'normal',
        'pc': 'normal', 'pcc': 'notpresent', 'ba': 'notpresent',
        'bgr': 120, 'bu': 40, 'sc': 1.2, 'sod': 135, 'pot': 4.5,
        'hemo': 12, 'pcv': 40, 'wbcc': 8000, 'rbcc': 4.5,
        'htn': 'no', 'dm': 'no', 'cad': 'no',
        'appet': 'good', 'pe': 'no', 'ane': 'no'
    }
    print("----------------Clinical data diagnosis started----------------\n")
    result = engine.run_diagnosis(test_patient)
    print(result)