# =================================================================
# PROJECT: Real-Time Financial Fraud Detection Engine
# ENGINEER: Shuvankar Choudhury and Tanya Pati
# PURPOSE: Real-time Consumer utilizing the PyTorch Hybrid Ensemble.
# =================================================================

import time
import pandas as pd
from hybrid_inference import HybridInferenceEngine
import warnings

warnings.filterwarnings('ignore')

# Advanced Terminal UI Colors
GREEN = '\033[92m'
RED = '\033[91m'
CYAN = '\033[96m'
RESET = '\033[0m'

def start_hybrid_security_scan():
    print(f"{CYAN}--- 🛡️ DEPLOYING HYBRID ENSEMBLE SECURITY LAYER ---{RESET}")
    
    # Initialize the new Hybrid Brain (PyTorch + GB + RF)
    engine = HybridInferenceEngine()
    
    # Load stream data (We write the raw pandas code here to keep it simple)
    df = pd.read_csv('data/creditcard.csv')
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df['Amount_Scaled'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df = df.drop(['Time', 'Amount'], axis=1)
    
    X_stream = df.drop(['Class'], axis=1)
    
    print(f"\n{GREEN}🟢 HYBRID SCANNER ACTIVE. Intercepting Live Stream...{RESET}\n")
    
    for index, row in X_stream.iterrows():
        transaction_id = f"TRX-{100000 + index}"
        features = row.values.reshape(1, -1)
        
        # ⚡ INFERENCE: Three models are now analyzing this simultaneously
        start_time = time.time()
        prediction = engine.predict(features)
        latency = (time.time() - start_time) * 1000 # Calculate latency in ms
        
        if prediction == 1:
            print(f"{RED}🚨 [BLOCK] {transaction_id} | HYBRID CONSENSUS: FRAUD DETECTED | Latency: {latency:.2f}ms{RESET}")
            time.sleep(1)
        else:
            print(f"{GREEN}✅ [PASS] {transaction_id} | Status: Verified | Latency: {latency:.2f}ms{RESET}")
            time.sleep(0.02)
            
        if index == 700: 
            print(f"\n{CYAN}--- 🛑 Diagnostic Complete ---{RESET}")
            break

if __name__ == "__main__":
    start_hybrid_security_scan()