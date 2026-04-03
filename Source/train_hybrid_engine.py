# =================================================================
# PROJECT: Real-Time Financial Fraud Detection Engine
# ENGINEER: Shuvankar Choudhury and Tanya Pati
# PURPOSE: Train a PyTorch + Gradient Boosting + Random Forest Hybrid Ensemble
# =================================================================

import pandas as pd
import numpy as np
import joblib
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from data_processor import load_and_clean_data
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. Define the PyTorch Deep Learning Architecture
# ---------------------------------------------------------
class DeepFraudNet(nn.Module):
    def __init__(self, input_shape):
        super(DeepFraudNet, self).__init__()
        # A 3-layer neural network designed to catch complex anomaly patterns
        self.network = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.ReLU(),
            nn.Dropout(0.2), # Prevents the AI from just memorizing the data
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # Outputs a probability between 0 (Safe) and 1 (Fraud)
        )

    def forward(self, x):
        return self.network(x)

def train_enterprise_ensemble():
    print("--- 🧠 INITIATING ENTERPRISE HYBRID AI BUILD ---")
    
    # Load Data
    df = load_and_clean_data('data/creditcard.csv')
    if df is None: return
    
    X = df.drop(['Class'], axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ---------------------------------------------------------
    # 2. Train the Scikit-Learn Models (The Traditional Experts)
    # ---------------------------------------------------------
    print("\n[1/3] Training Gradient Boosting Engine...")
    gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gb_model.fit(X_train, y_train)

    print("[2/3] Training Random Forest Engine...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # 3. Train the PyTorch Deep Learning Model
    # ---------------------------------------------------------
    print("[3/3] Compiling PyTorch Deep Learning Network...")
    
    # Convert Pandas data to PyTorch Tensors (Math matrices)
    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)
    
    # Initialize the Neural Network
    pytorch_model = DeepFraudNet(X_train.shape[1])
    criterion = nn.BCELoss() # Binary Cross Entropy (Standard for 0/1 classification)
    optimizer = optim.Adam(pytorch_model.parameters(), lr=0.001)

    print("      >> Training PyTorch Tensors (5 Epochs)...")
    pytorch_model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = pytorch_model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # ---------------------------------------------------------
    # 4. Save the Hybrid Ensemble
    # ---------------------------------------------------------
    if not os.path.exists('models'):
        os.makedirs('models')
        
    # We save all three brains together in a dictionary
    ensemble = {
        'gradient_boosting': gb_model,
        'random_forest': rf_model,
        'pytorch_state_dict': pytorch_model.state_dict(),
        'input_shape': X_train.shape[1]
    }
    
    joblib.dump(ensemble, 'models/hybrid_ensemble.pkl')
    print("\n✅ SYSTEM ARCHITECTURE: Hybrid PyTorch + GB + RF model successfully saved to models/hybrid_ensemble.pkl")

if __name__ == "__main__":
    train_enterprise_ensemble()