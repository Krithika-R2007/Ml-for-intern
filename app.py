from flask import Flask, request, jsonify, render_template
import os
import json
import joblib
import numpy as np

app = Flask(__name__)

# Paths for files
MODEL_PATH = 'best_model.pkl'
SCALER_PATH = 'scaler.pkl'
METRICS_PATH = 'evaluation_results.json'

def load_ml_resources():
    """Load model, scaler, and metrics from disk if they exist."""
    model = None
    scaler = None
    metrics = None
    
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
            
    return model, scaler, metrics

@app.route('/')
def home():
    # Render index.html. Flask looks inside 'templates/' folder
    return render_template('index.html')

@app.route('/metrics', methods=['GET'])
def get_metrics():
    _, _, metrics = load_ml_resources()
    if metrics is None:
        return jsonify({'error': 'Models have not been trained yet. Please run training pipeline.'}), 404
    return jsonify(metrics)

@app.route('/predict', methods=['POST'])
def predict():
    model, scaler, metrics = load_ml_resources()
    
    if model is None or scaler is None or metrics is None:
        return jsonify({
            'error': 'Prediction models are not ready. Please run the training pipeline first.'
        }), 400
        
    try:
        data = request.get_json()
        
        # Extract features in the correct order as defined during training
        features = metrics.get('features', [
            'previous_donation_amount',
            'donation_frequency',
            'time_since_last_donation',
            'campaign_participation_history'
        ])
        
        input_data = []
        for feature in features:
            if feature not in data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400
            input_data.append(float(data[feature]))
            
        # Reshape for prediction
        input_array = np.array(input_data).reshape(1, -1)
        
        # Scale features
        scaled_input = scaler.transform(input_array)
        
        # Predict class and probability
        prediction = int(model.predict(scaled_input)[0])
        prob = float(model.predict_proba(scaled_input)[:, 1][0])
        
        response = {
            'success': True,
            'prediction': 'Yes' if prediction == 1 else 'No',
            'probability': round(prob * 100, 1),
            'model_used': metrics.get('best_model', 'Unknown')
        }
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting NayePankh Foundation prediction server...")
    # Run server on port 5000
    app.run(debug=True, host='127.0.0.1', port=5000)
