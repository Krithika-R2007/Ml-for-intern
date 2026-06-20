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
        # Fill numeric values with median, and categorical with mode
        numeric_cols = df_ml.select_dtypes(include=[np.number]).columns
        df_ml[numeric_cols] = df_ml[numeric_cols].fillna(df_ml[numeric_cols].median())
        categorical_cols = df_ml.select_dtypes(exclude=[np.number]).columns
        for col in categorical_cols:
            df_ml[col] = df_ml[col].fillna(df_ml[col].mode()[0])
        
    # Check for target columns
    retention_target = 'will_donate_again'
    channel_target = 'preferred_channel'
    if retention_target not in df_ml.columns:
        raise ValueError(f"Retention target '{retention_target}' not found in dataset. Existing columns: {list(df_ml.columns)}")
    if channel_target not in df_ml.columns:
        raise ValueError(f"Channel target '{channel_target}' not found in dataset. Existing columns: {list(df_ml.columns)}")
        
    # Set up features list
    feature_names = [col for col in df_ml.columns if col not in [retention_target, channel_target]]
    X = df_ml[feature_names]
    y_ret = df_ml[retention_target]
    
    # Channel target map to integers
    channels_map = {'Email': 0, 'WhatsApp': 1, 'SMS': 2, 'Phone Call': 3, 'Social Media': 4}
    y_chan = df_ml[channel_target].map(channels_map)
    
    print(f"Features for training: {feature_names}")
    print(f"Targets: {retention_target} (binary), {channel_target} (multiclass)")
    
    # 2. Train-Test Split (80% train, 20% test)
    # We perform split using indices so we have aligned train/test sets for both targets
    indices = np.arange(df_ml.shape[0])
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y_ret)
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train_ret, y_test_ret = y_ret.iloc[train_idx], y_ret.iloc[test_idx]
    y_train_chan, y_test_chan = y_chan.iloc[train_idx], y_chan.iloc[test_idx]
    
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # 3. Scaling features (both models share the same feature space)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    scaler_path = 'scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"Saved feature scaler to {scaler_path}")
    
    # 4. Train Retention Models
    print("\n--- Training Retention Predictor ---")
    print("Training Logistic Regression...")
    lr_ret = LogisticRegression(random_state=42, max_iter=1000)
    lr_ret.fit(X_train_scaled, y_train_ret)
    
    lr_ret_preds = lr_ret.predict(X_test_scaled)
    lr_ret_probs = lr_ret.predict_proba(X_test_scaled)[:, 1]
    
    lr_ret_metrics = {
        'accuracy': float(accuracy_score(y_test_ret, lr_ret_preds)),
        'precision': float(precision_score(y_test_ret, lr_ret_preds)),
        'recall': float(recall_score(y_test_ret, lr_ret_preds)),
        'f1': float(f1_score(y_test_ret, lr_ret_preds)),
        'roc_auc': float(roc_auc_score(y_test_ret, lr_ret_probs))
    }
    print(f"Logistic Regression Retention Metrics: {lr_ret_metrics}")
    
    print("Training Random Forest Classifier...")
    rf_ret = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_ret.fit(X_train_scaled, y_train_ret)
    
    rf_ret_preds = rf_ret.predict(X_test_scaled)
    rf_ret_probs = rf_ret.predict_proba(X_test_scaled)[:, 1]
    
    rf_ret_metrics = {
        'accuracy': float(accuracy_score(y_test_ret, rf_ret_preds)),
        'precision': float(precision_score(y_test_ret, rf_ret_preds)),
        'recall': float(recall_score(y_test_ret, rf_ret_preds)),
        'f1': float(f1_score(y_test_ret, rf_ret_preds)),
        'roc_auc': float(roc_auc_score(y_test_ret, rf_ret_probs))
    }
    print(f"Random Forest Retention Metrics: {rf_ret_metrics}")
    
    # Extract Feature Importances/Coefficients for Retention
    rf_ret_importances = {name: float(imp) for name, imp in zip(feature_names, rf_ret.feature_importances_)}
    lr_ret_coefs = {name: float(coef) for name, coef in zip(feature_names, lr_ret.coef_[0])}
    
    # Select Best Retention Model
    if rf_ret_metrics['f1'] >= lr_ret_metrics['f1']:
        best_ret_name = "Random Forest"
        best_ret_model = rf_ret
        best_ret_metrics = rf_ret_metrics
    else:
        best_ret_name = "Logistic Regression"
        best_ret_model = lr_ret
        best_ret_metrics = lr_ret_metrics
        
    print(f"Selected Best Retention Model: {best_ret_name}")
    joblib.dump(best_ret_model, 'best_model.pkl')
    
    # 5. Train Channel Predictor Models (Multiclass)
    print("\n--- Training Campaign Channel Predictor ---")
    print("Training Multiclass Logistic Regression...")
    lr_chan = LogisticRegression(random_state=42, max_iter=1000, multi_class='multinomial')
    lr_chan.fit(X_train_scaled, y_train_chan)
    
    lr_chan_preds = lr_chan.predict(X_test_scaled)
    
    lr_chan_metrics = {
        'accuracy': float(accuracy_score(y_test_chan, lr_chan_preds)),
        'precision': float(precision_score(y_test_chan, lr_chan_preds, average='macro', zero_division=0)),
        'recall': float(recall_score(y_test_chan, lr_chan_preds, average='macro', zero_division=0)),
        'f1': float(f1_score(y_test_chan, lr_chan_preds, average='macro', zero_division=0))
    }
    print(f"Logistic Regression Channel Metrics: {lr_chan_metrics}")
    
    print("Training Random Forest Channel Classifier...")
    rf_chan = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_chan.fit(X_train_scaled, y_train_chan)
    
    rf_chan_preds = rf_chan.predict(X_test_scaled)
    
    rf_chan_metrics = {
        'accuracy': float(accuracy_score(y_test_chan, rf_chan_preds)),
        'precision': float(precision_score(y_test_chan, rf_chan_preds, average='macro', zero_division=0)),
        'recall': float(recall_score(y_test_chan, rf_chan_preds, average='macro', zero_division=0)),
        'f1': float(f1_score(y_test_chan, rf_chan_preds, average='macro', zero_division=0))
    }
    print(f"Random Forest Channel Metrics: {rf_chan_metrics}")
    
    # Extract Feature Importances/Coefficients for Channel
    rf_chan_importances = {name: float(imp) for name, imp in zip(feature_names, rf_chan.feature_importances_)}
    # Average absolute coefficient across classes for Logistic Regression
    lr_chan_coefs = {name: float(np.mean(np.abs(lr_chan.coef_[:, idx]))) for idx, name in enumerate(feature_names)}
    
    # Select Best Channel Model
    if rf_chan_metrics['f1'] >= lr_chan_metrics['f1']:
        best_chan_name = "Random Forest"
        best_chan_model = rf_chan
        best_chan_metrics = rf_chan_metrics
    else:
        best_chan_name = "Logistic Regression"
        best_chan_model = lr_chan
        best_chan_metrics = lr_chan_metrics
        
    print(f"Selected Best Channel Model: {best_chan_name}")
    joblib.dump(best_chan_model, 'channel_model.pkl')
    
    # 6. Save channel label encoding mapping alongside the model
    joblib.dump(channels_map, 'channels_map.pkl')
    
    # 7. Export evaluation details for dashboard
    results = {
        'best_model': best_ret_name,
        'features': feature_names,
        'models': {
            'Logistic Regression': {
                'metrics': lr_ret_metrics,
                'insights': lr_ret_coefs
            },
            'Random Forest': {
                'metrics': rf_ret_metrics,
                'insights': rf_ret_importances
            }
        },
        'channel_model': {
            'best_model': best_chan_name,
            'models': {
                'Logistic Regression': {
                    'metrics': lr_chan_metrics,
                    'insights': lr_chan_coefs
                },
                'Random Forest': {
                    'metrics': rf_chan_metrics,
                    'insights': rf_chan_importances
                }
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
