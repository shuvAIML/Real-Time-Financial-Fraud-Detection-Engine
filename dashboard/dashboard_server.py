# =================================================================
# PROJECT: Real-Time Financial Fraud Detection Engine
# ENGINEER: Shuvankar Choudhury and Tanya Pati
# PURPOSE: Web Server connecting the Hybrid AI to the UI Dashboard.
# =================================================================

from flask import Flask, render_template
from flask_socketio import SocketIO
import time
import sys
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Safely point Python to your 'Source' folder so it can find your AI script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Source')))
from hybrid_inference import HybridInferenceEngine

app = Flask(__name__)
socketio = SocketIO(app)

# Load the PyTorch Hybrid AI Brain
print("[SERVER] Loading Hybrid Architecture from disk...")
try:
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'hybrid_ensemble.pkl'))
    engine = HybridInferenceEngine(model_path)
    print("[SERVER] AI successfully loaded.")
except Exception as e:
    print(f"ERROR: Cannot load AI. {e}")
    sys.exit()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start_stream')
def handle_stream():
    print("[SERVER] UI requested stream start. Feeding data to AI...")
    
    # 1. Load the raw data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'creditcard.csv'))
    df = pd.read_csv(data_path)
    ui_amounts = df['Amount'].copy() # Keep raw amounts for the UI display
    
    # 2. Prepare data for AI
    scaler = StandardScaler()
    df['Amount_Scaled'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df = df.drop(['Time', 'Amount'], axis=1)
    X_stream = df.drop(['Class'], axis=1)
    
    # 3. Process transactions
    for index, row in X_stream.iterrows():
        transaction_id = f"TRX-{100000 + index}"
        features = row.values.reshape(1, -1)
        
        # ⚡ INFERENCE
        prediction = engine.predict(features)
        display_amount = round(ui_amounts.iloc[index], 2)
        
        if prediction == 1: # FRAUD
            socketio.emit('transaction_update', {'id': transaction_id, 'status': 'FRAUD', 'amount': display_amount})
            time.sleep(1.5) # Dramatic pause on UI
        else: # SAFE
            socketio.emit('transaction_update', {'id': transaction_id, 'status': 'SAFE', 'amount': display_amount})
            time.sleep(0.05) # Fast scan
            
        if index == 700:
            break

if __name__ == '__main__':
    print("[SERVER] Initializing UI Dashboard on port 5000...")
    socketio.run(app, debug=True, port=5000, allow_unsafe_werkzeug=True)
