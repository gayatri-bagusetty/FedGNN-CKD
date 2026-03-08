import joblib
import torch
import os
import sys
import numpy as np
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import ModelMode

# PATH SETUP
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

from data.models.gcn_model import GCN
from notebooks.preprocessing import preprocess_single_patient
from notebooks.graph_construction import build_patient_similarity_graph
from notebooks.inference_xai import generate_clinical_description

# Wrapper model for GNNExplainer
class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x, edge_index):
        logits = self.model(x, edge_index)
        probs = torch.softmax(logits, dim=1)
        return probs[:, 1]  # CKD probability

# Clinical Inference Engine
class ClinicalInferenceEngine:
    def __init__(self, model_path, scaler_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler_path = scaler_path
        self.feature_names = joblib.load(
            os.path.join(BASE_DIR, "data", "processed", "feature_order.pkl")
        )
        graph_path = os.path.join(BASE_DIR, "data", "graph", "hospital_A_train.pt")
        graph = torch.load(graph_path)
        self.train_features = graph.x.cpu().numpy()
        self.edge_index = graph.edge_index
        self.train_edge_index = torch.load(
            os.path.join(BASE_DIR, "data", "processed", "edge_index.pt")
        )

        # Load Global Federated Model
        self.model = GCN(
            input_dim=24,
            hidden_dim=32,
            output_dim=2
        ).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.eval()

        # XAI Wrapper
        self.wrapped_model = WrappedModel(self.model)
        self.explainer = Explainer(
            model=self.wrapped_model,
            algorithm=GNNExplainer(epochs=100),
            explanation_type="model",
            node_mask_type="attributes",
            model_config=dict(
                mode=ModelMode.binary_classification,
                task_level="node",
                return_type="probs"
            )
        )

    # MAIN DIAGNOSIS PIPELINE
    def run_diagnosis(self, raw_patient_data):
        # 1. Preprocess patient
        x_scaled = preprocess_single_patient(
            raw_patient_data
        )
        x_tensor = torch.tensor(
            x_scaled,
            dtype=torch.float32
        )

        # 2. Build graph
        x, edge_index, new_node = build_patient_similarity_graph(
            x_tensor,
            self.train_features,
            self.train_edge_index,
            k=5
        )
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)

        # 3. Prediction
        with torch.no_grad():
            logits = self.model(x, edge_index)
            probs = torch.softmax(logits, dim=1)
            print("Model logits:", logits)
            print("Model probabilities:", probs)
            prob_ckd = probs[new_node][1].item()
            print("Raw CKD probability:", prob_ckd)
            
        # Clinical Probability Adjustment
        creatinine = float(raw_patient_data["sc"])
        urea = float(raw_patient_data["bu"])
        hemo = float(raw_patient_data["hemo"])

        adjustment = 0.0
        if creatinine > 1.3:
            adjustment += 0.05
        if urea > 45:
            adjustment += 0.05
        if hemo < 11:
            adjustment += 0.03
        print("Clinical adjustment:", adjustment)
        prob_ckd = min(prob_ckd + adjustment, 1.0)
        print("Adjusted CKD probability:", prob_ckd)

        # Final Decision
        prediction = (
            "CKD" if prob_ckd >= 0.7
            else "Borderline Risk" if prob_ckd >= 0.4
            else "Non-CKD"
        )

        # 4. Explainability
        torch.manual_seed(42)
        np.random.seed(42)
        explanation = self.explainer(x, edge_index, index=new_node)
        importances = (
            explanation.node_mask
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )

        feat_imp = sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )

        # 5. Clinical explanation
        report = generate_clinical_description(
            prediction,
            prob_ckd,
            feat_imp
        )
        
        # 6. Final Output
        return {
            "prediction": prediction,
            "probability": prob_ckd,
            "risk_score": prob_ckd,
            "top_features": feat_imp[:5],
            "clinical_explanation": report["clinical_explanation"],
            "recommendation": report["recommendation"],
            "primary_biomarkers": report["primary_biomarkers"],
            "chart_values": report["chart_values"]
        }