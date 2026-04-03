# =================================================================
# PROJECT: Real-Time Financial Fraud Detection Engine
# ENGINEER: Shuvankar Choudhury and Tanya Pati
# PURPOSE: Hybrid Inference Logic using PyTorch + GB + RF Ensemble
# =================================================================

import torch
import torch.nn as nn
import joblib
import numpy as np

# Re-defining the PyTorch architecture so we can load the weights
class DeepFraudNet(nn.Module):
    def __init__(self, input_shape):
        super(DeepFraudNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

class HybridInferenceEngine:
    def __init__(self, model_path='models/hybrid_ensemble.pkl'):
        print("[SYSTEM] Initializing Hybrid Inference Engine...")
        
        # 1. Load the ensemble dictionary we saved during training
        ensemble = joblib.load(model_path)
        
        self.rf_model = ensemble['random_forest']
        self.gb_model = ensemble['gradient_boosting']
        
        # 2. Reconstruct and load the PyTorch weights
        self.pytorch_model = DeepFraudNet(ensemble['input_shape'])
        self.pytorch_model.load_state_dict(ensemble['pytorch_state_dict'])
        self.pytorch_model.eval() # Set to evaluation mode (turns off Dropout)

    def predict(self, transaction_features):
        """
        An advanced ensemble voting mechanism to ensure maximum precision.
        """
        # Convert data for PyTorch
        tensor_data = torch.FloatTensor(transaction_features)
        
        # Get individual predictions
        # Probability from PyTorch (Deep Learning)
        with torch.no_grad():
            torch_prob = self.pytorch_model(tensor_data).item()
        
        # Predictions from Scikit-Learn models
        rf_pred = self.rf_model.predict(transaction_features)[0]
        gb_pred = self.gb_model.predict(transaction_features)[0]
        
        # --- ENSEMBLE VOTING LOGIC ---
        # We flag as FRAUD (1) only if at least two models agree, 
        # or if the PyTorch model is extremely confident (prob > 0.95).
        votes = rf_pred + gb_pred + (1 if torch_prob > 0.5 else 0)
        
        if votes >= 2 or torch_prob > 0.95:
            return 1 # FRAUD
        return 0 # SAFE

if __name__ == "__main__":
    # Quick Test with dummy data
    engine = HybridInferenceEngine()
    print("Hybrid Engine Ready for Stream Processing.")