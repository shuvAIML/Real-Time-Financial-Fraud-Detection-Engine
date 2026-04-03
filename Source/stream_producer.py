# =================================================================
# PROJECT: Real-Time Financial Fraud Detection Engine
# ENGINEER: Shuvankar Choudhury and Tanya Pati
# PURPOSE: Simulate a high-throughput live data stream (Bank Terminal).
# =================================================================

import pandas as pd
import time
import json

# In a full production environment, this would connect to Apache Kafka.
# For this local architecture, we are simulating the continuous data feed.

def start_live_stream(file_path):
    print("--- 📡 Initializing Live Bank Feed ---")
    print("Connecting to secure transaction network...\n")
    time.sleep(2)
    
    try:
        # Load the data, but we will stream it line by line
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: Could not locate the transaction database.")
        return

    print("🟢 STATUS: LIVE. Broadcasting transactions...\n")
    print("-" * 50)
    
    # Simulating the live stream of data
    for index, row in df.iterrows():
        # Convert the row to a JSON dictionary (Industry standard for web streaming)
        transaction_data = row.to_dict()
        
        # This is where we would send data to Kafka. 
        # For now, we print it to the console to visualize the stream.
        transaction_id = f"TRX-{100000 + index}"
        amount = transaction_data['Amount']
        
        print(f"[STREAM] {transaction_id} | Amount: ${amount:.2f} | Routing to Security Engine...")
        
        # Simulate network latency (0.1 seconds between swipes)
        # We keep this fast to show 'high-throughput' capabilities.
        time.sleep(0.1)
        
        # Stop after 50 transactions just for our visual test
        if index == 50:
            print("\n--- 🛑 Stream Test Paused ---")
            break

if __name__ == "__main__":
    start_live_stream('data/creditcard.csv')