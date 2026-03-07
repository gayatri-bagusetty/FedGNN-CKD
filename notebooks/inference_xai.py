import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import shap
import os
import sys
from sklearn.preprocessing import StandardScaler

# PATH SETUP
sys.path.append(os.path.abspath(".."))
from data.models.gcn_model import GCN

plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 12

# CKD STAGES
CKD_STAGES = ['Stage 0', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']

def create_background_dataset(n_samples=100):
    """Generate background for SHAP"""
    # Simulate 100 patients (your 42 features)
    np.random.seed(42)
    background = np.random.normal(0, 1, (n_samples, 42)).astype(np.float32)
    # Add realistic CKD patterns to background
    for i in range(n_samples):
        if i % 20 == 0:  # 5% severe CKD in background
            background[i, 11] += 3  # SC (serum creatinine)
            background[i, 3] += 2   # AL (albumin)
            background[i, 14] -= 2  # HEMO (hemoglobin)
    return torch.FloatTensor(background)

def run_shap_xai():
    """SHAP + Captum XAI for FedGNN-CKD"""
    
    # YOUR MODEL DIMENSIONS + GLOBAL LDP
    model = GCN(input_dim=42, hidden_dim=128, output_dim=5)
    model_path = "../data/models/global_model_ldp.pth"
    
    print("Loading FedGNN-CKD Global LDP Model...")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    # Test patients (Stage 4 CKD, Stage 2 CKD, Healthy)
    test_patients = np.array([
        # Stage 4 CKD: High SC, AL, low HEMO
        [65, 160, 1.005, 4, 2, 0, 1, 1, 1, 250, 85, 7.2, 128, 6.2, 7.5, 22, 12000, 2.8, 1, 1, 1, 0, 1, 1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        # Stage 2 CKD: Moderate SC elevation
        [55, 140, 1.015, 2, 0, 0, 0, 0, 0, 180, 45, 2.1, 135, 4.8, 11.2, 35, 9000, 4.2, 1, 0, 0, 1, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        # Healthy: Normal biomarkers
        [35, 120, 1.020, 0, 0, 1, 0, 0, 0, 95, 18, 0.9, 140, 4.2, 15.0, 45, 7500, 5.0, 0, 0, 0, 1, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    ], dtype=np.float32)
    
    # Feature names (first 24 meaningful)
    feature_names = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr',
                    'bu', 'SC', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 
                    'htn', 'dm', 'cad', 'appet', 'pe', 'ane'] + [f'f{i}' for i in range(18)]
    
    # SHAP KernelExplainer (model-agnostic)
    print("Computing SHAP values...")
    background = create_background_dataset()
    
    # Wrapper for SHAP (single forward pass)
    def model_predict(X):
        X_tensor = torch.FloatTensor(X).unsqueeze(1)  # [N, 1, 42]
        edge_index = torch.tensor([[0]], dtype=torch.long).repeat(1, X.shape[0]).t()
        with torch.no_grad():
            logits = model(X_tensor[:, 0], edge_index)
            return logits.cpu().numpy()
    
    # SHAP Explainer
    explainer = shap.KernelExplainer(model_predict, background.numpy())
    shap_values = explainer.shap_values(test_patients, nsamples=100)
    
    # Predictions
    predictions = model_predict(test_patients)
    probs = F.softmax(torch.FloatTensor(predictions), dim=1).numpy()
    pred_classes = np.argmax(probs, axis=1)
    
    print("\nSHAP XAI Results:")
    for i, (pred, prob, patient) in enumerate(zip(pred_classes, probs, test_patients)):
        stage = CKD_STAGES[pred]
        print(f"\nPatient {i+1}: {stage} ({prob[pred]:.1%} confidence)")
        print(f"  Top drivers: SC={shap_values[pred][i][11]:+.3f}, AL={shap_values[pred][i][3]:+.3f}")
    
    # === PUBLICATION PLOTS ===
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Summary plot (SHAP standard)
    plt.subplot(2, 3, 1)
    shap.summary_plot(shap_values[pred_classes[0]], test_patients, feature_names=feature_names[:24],
                     max_display=12, show=False)
    plt.title(f'SHAP Summary - {CKD_STAGES[pred_classes[0]]}', fontsize=14, fontweight='bold')
    
    # Plot 2: Patient-specific bar
    plt.subplot(2, 3, 2)
    top_features = np.argsort(np.abs(shap_values[pred_classes[0]][0]))[-10:]
    plt.barh(range(10), shap_values[pred_classes[0]][0, top_features][::-1])
    plt.yticks(range(10), [feature_names[i][:10] for i in top_features[::-1]])
    plt.xlabel('SHAP Value')
    plt.title('Patient 1 Feature Importance')
    
    # Plot 3: Force plot (Patient 1)
    plt.subplot(2, 3, 3)
    shap.force_plot(explainer.expected_value[pred_classes[0]], 
                   shap_values[pred_classes[0]][0], test_patients[0],
                   feature_names=feature_names[:24], matplotlib=True, show=False)
    plt.title('SHAP Force Plot')
    
    # Plot 4: Multi-class contribution
    plt.subplot(2, 3, 4)
    for i, stage in enumerate(CKD_STAGES):
        plt.scatter(test_patients[:, 11], shap_values[i][:, 11], label=stage, s=100, alpha=0.7)
    plt.xlabel('Serum Creatinine (SC)')
    plt.ylabel('SHAP value for SC')
    plt.title('SC Contribution by CKD Stage')
    plt.legend()
    
    # Plot 5: Waterfall plot
    plt.subplot(2, 3, 5)
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[pred_classes[0]][0],
        base_values=explainer.expected_value[pred_classes[0]],
        data=test_patients[0],
        feature_names=feature_names[:24]
    ), show=False)
    
    # Plot 6: Model prediction vs SHAP
    plt.subplot(2, 3, 6)
    x_pos = np.arange(5)
    plt.bar(x_pos, probs[0], color='coral', alpha=0.8, label='Model Prediction')
    plt.plot(x_pos[pred_classes[0]], probs[0, pred_classes[0]], 'ro', markersize=12, label='Predicted')
    plt.xticks(x_pos, CKD_STAGES)
    plt.ylabel('Probability')
    plt.title('5-Class CKD Prediction')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('../data/plots/shap_fedgnn_ckd.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nSHAP XAI COMPLETE!")
    print("Plot saved: shap_fedgnn_ckd.png")
    
    return shap_values, probs

if __name__ == "__main__":
    # Install: pip install shap
    shap_values, probs = run_shap_xai()