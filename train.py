import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_and_evaluate(filepath=None):
    if filepath is None:
        if os.path.exists('donor_data_inr.csv'):
            filepath = 'donor_data_inr.csv'
        elif os.path.exists('donor_data.csv'):
            filepath = 'donor_data.csv'
        else:
            filepath = 'donor_data_inr.csv'
            
    print(f"Loading dataset from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}. Please generate or add it first.")
        
    df = pd.read_csv(filepath)
    
    # 1. Basic Cleaning
    # If there are identification columns like 'donor_id' or 'name', we drop them for training
    id_cols = ['donor_id', 'name']
    cols_to_drop = [col for col in id_cols if col in df.columns]
    if cols_to_drop:
        print(f"Dropping columns not useful for machine learning: {cols_to_drop}")
        df_ml = df.drop(columns=cols_to_drop)
    else:
        df_ml = df.copy()
        
    # Handle missing values if any
    if df_ml.isnull().sum().sum() > 0:
        print("Handling missing values (filling with median)...")
        df_ml = df_ml.fillna(df_ml.median())
        
    # Check for target column
    target_col = 'will_donate_again'
    if target_col not in df_ml.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset. Existing columns: {list(df_ml.columns)}")
        
    X = df_ml.drop(columns=[target_col])
    y = df_ml[target_col]
    
    feature_names = list(X.columns)
    print(f"Features for training: {feature_names}")
    print(f"Target variable: {target_col}")
    
    # 2. Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # 3. Scaling features (especially important for Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    scaler_path = 'scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"Saved feature scaler to {scaler_path}")
    
    # 4. Train Logistic Regression
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    
    # Predict & evaluate
    lr_preds = lr_model.predict(X_test_scaled)
    lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    lr_metrics = {
        'accuracy': float(accuracy_score(y_test, lr_preds)),
        'precision': float(precision_score(y_test, lr_preds)),
        'recall': float(recall_score(y_test, lr_preds)),
        'f1': float(f1_score(y_test, lr_preds)),
        'roc_auc': float(roc_auc_score(y_test, lr_probs))
    }
    print(f"Logistic Regression Metrics: {lr_metrics}")
    
    # 5. Train Random Forest Classifier
    print("\nTraining Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Predict & evaluate
    rf_preds = rf_model.predict(X_test_scaled)
    rf_probs = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    rf_metrics = {
        'accuracy': float(accuracy_score(y_test, rf_preds)),
        'precision': float(precision_score(y_test, rf_preds)),
        'recall': float(recall_score(y_test, rf_preds)),
        'f1': float(f1_score(y_test, rf_preds)),
        'roc_auc': float(roc_auc_score(y_test, rf_probs))
    }
    print(f"Random Forest Metrics: {rf_metrics}")
    
    # 6. Extract Feature Importances / Coefficients
    # Random Forest importances
    rf_importances = {name: float(imp) for name, imp in zip(feature_names, rf_model.feature_importances_)}
    # Logistic Regression coefficients (absolute values represent importance)
    lr_coefs = {name: float(coef) for name, coef in zip(feature_names, lr_model.coef_[0])}
    
    # 7. Select and Save the Best Model
    # We will use F1-score as the selector metric, falling back to Accuracy
    if rf_metrics['f1'] >= lr_metrics['f1']:
        best_model_name = "Random Forest"
        best_model = rf_model
        best_metrics = rf_metrics
    else:
        best_model_name = "Logistic Regression"
        best_model = lr_model
        best_metrics = lr_metrics
        
    print(f"\nSelected Best Model: {best_model_name}")
    model_path = 'best_model.pkl'
    joblib.dump(best_model, model_path)
    print(f"Saved best model to {model_path}")
    
    # 8. Export evaluation details for dashboard to read
    results = {
        'best_model': best_model_name,
        'features': feature_names,
        'models': {
            'Logistic Regression': {
                'metrics': lr_metrics,
                'insights': lr_coefs
            },
            'Random Forest': {
                'metrics': rf_metrics,
                'insights': rf_importances
            }
        }
    }
    
    results_path = 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved evaluation insights to {results_path}")
    print("Training pipeline complete.")

if __name__ == '__main__':
    train_and_evaluate()
