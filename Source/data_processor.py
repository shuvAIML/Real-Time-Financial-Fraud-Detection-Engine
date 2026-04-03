# =================================================================
# PROJECT: Real-Time Financial Fraud Detection Engine
# ENGINEER: Shuvankar Choudhury and Tanya Pati
# PURPOSE: Prepare raw transaction data for high-precision AI training.
# =================================================================

import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_clean_data(file_path):
    print("--- 🛠️ Starting Data Preprocessing ---")
    
    try:
        # Load the raw dataset from the 'data' folder
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Check your 'data' folder.")
        return None

    # Feature Engineering: Scaling the transaction 'Amount'
    # This prevents the AI from being biased by massive transactions.
    scaler = StandardScaler()
    df['Amount_Scaled'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    
    # Drop columns that do not contribute to the anomaly detection logic
    df = df.drop(['Time', 'Amount'], axis=1)
    
    print(f"Success! Processed {len(df)} transactions.")
    return df

if __name__ == "__main__":
    # Test run (Note: '..' moves up one level from 'Source' to find 'data')
    data = load_and_clean_data('data/creditcard.csv')