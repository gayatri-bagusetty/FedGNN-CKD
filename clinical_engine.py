import torch
import os
import sys
import numpy as np
import torch.nn.functional as F
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import ModelMode

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))
sys.path.append(BASE_DIR)

from data.models.gcn_model import GCN
from notebooks.preprocessing import preprocess_single_patient
from notebooks.graph_construction import build_single_node_graph
from notebooks.inference_xai import generate_clinical_description


# ------------------------------------------------------
# Wrapper model for GNNExplainer
# ------------------------------------------------------
class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index):
        out = self.model(x, edge_index)
        return torch.softmax(out, dim=1)[:, 1]


# ------------------------------------------------------
# Clinical Inference Engine
# ------------------------------------------------------
class ClinicalInferenceEngine:

    def __init__(self, model_path, scaler_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler_path = scaler_path

        self.feature_names = [
            'age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu',
            'sc','sod','pot','hemo','pcv','wbcc','rbcc','htn','dm','cad',
            'appet','pe','ane'
        ]

        # Load global federated model
        self.model = GCN(
            input_dim=24,
            hidden_dim=32,
            output_dim=2
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )

        self.model.eval()

        # Wrapped model for XAI
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

    # ------------------------------------------------------
    # MAIN PIPELINE
    # ------------------------------------------------------
    def run_diagnosis(self, raw_patient_data):

        # --------------------------------------------------
        # 1. Preprocessing
        # --------------------------------------------------
        x_scaled = preprocess_single_patient(
            raw_patient_data,
            self.scaler_path
        )

        x_tensor = torch.tensor(
            x_scaled,
            dtype=torch.float32
        ).to(self.device)

        # --------------------------------------------------
        # 2. Graph construction
        # --------------------------------------------------
        x, edge_index = build_single_node_graph(x_tensor)
        x, edge_index = x.to(self.device), edge_index.to(self.device)

        # --------------------------------------------------
        # 3. Local personalization (fine-tuning)
        # --------------------------------------------------
        self.model.train()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.0005
        )

        # lightweight personalization
        for _ in range(5):
            optimizer.zero_grad()
            out = self.model(x, edge_index)
            loss = F.cross_entropy(
                out,
                torch.tensor([1]).to(self.device)
            )
            loss.backward()
            optimizer.step()

        self.model.eval()

        # --------------------------------------------------
        # 4. Prediction
        # --------------------------------------------------
        with torch.no_grad():
            logits = self.model(x, edge_index)
            prob = torch.softmax(logits, dim=1)[0][1].item()

        prediction = "CKD" if prob >= 0.50 else "Non-CKD"

        # --------------------------------------------------
        # 5. Explainability (XAI)
        # --------------------------------------------------
        explanation = self.explainer(x, edge_index)

        importances = (
            explanation.node_mask.squeeze()
            .detach()
            .cpu()
            .numpy()
        )

        feat_imp = sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )

        # --------------------------------------------------
        # 6. Clinical explanation text
        # --------------------------------------------------
        report = generate_clinical_description(
            prediction,
            prob,
            feat_imp
        )

        # --------------------------------------------------
        # 7. Final output
        # --------------------------------------------------
        return {
            "prediction": prediction,
            "probability": prob,
            "top_features": feat_imp[:5],
            "report": report
        }