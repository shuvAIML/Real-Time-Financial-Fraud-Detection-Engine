# =================================================================
# PROJECT: Real-Time Financial Fraud Detection Engine
# ENGINEER: Shuvankar Choudhury and Tanya Pati
# PURPOSE: Train and save the Random Forest classification model.
# =================================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib 
import os

# Important: We import the cleaning function from our data_processor file
from data_processor import load_and_clean_data

def train_model():
    print("--- 🧠 Initializing the Guardian AI ---")
    
    # 1. Load and clean the data. 
    # Since we run this from the main folder, the path is simply 'data/...'
    df = load_and_clean_data('data/creditcard.csv')
    if df is None:
        return
    
    # 2. Separate Features (X) and Target Labels (y)
    X = df.drop(['Class'], axis=1)
    y = df['Class']
    
    # 3. Split into Training (80%) and Testing (20%) datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Initialize the AI (100 Decision Trees for high accuracy)
    # n_jobs=-1 uses all your CPU cores to make it faster.
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=1)
    
    # 5. Train the Model
    print("\nThe AI is currently studying transaction patterns... please wait.")
    model.fit(X_train, y_train)
    
    # 6. Evaluate Performance
    predictions = model.predict(X_test)
    print("\n--- 📊 Final Performance Report ---")
    print(classification_report(y_test, predictions))
    
    # 7. Ensure 'models' directory exists and save the trained "Brain"
    if not os.path.exists('models'):
        os.makedirs('models')
        
    joblib.dump(model, 'models/guardian_rf.pkl')
    print("\nSystem Architecture: Model successfully saved to models/guardian_rf.pkl")

if __name__ == "__main__":
    train_model()