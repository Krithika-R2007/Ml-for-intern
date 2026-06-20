from flask import Flask, request, jsonify, render_template
import os
import json
import joblib
import numpy as np

app = Flask(__name__)

# Paths for files
MODEL_PATH = 'best_model.pkl'
CHANNEL_MODEL_PATH = 'channel_model.pkl'
SCALER_PATH = 'scaler.pkl'
METRICS_PATH = 'evaluation_results.json'

def load_ml_resources():
    """Load models, scaler, and metrics from disk if they exist."""
    model = None
    channel_model = None
    scaler = None
    metrics = None
    
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    if os.path.exists(CHANNEL_MODEL_PATH):
        channel_model = joblib.load(CHANNEL_MODEL_PATH)
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
            
    return model, channel_model, scaler, metrics

@app.route('/')
def home():
    # Render index.html. Flask looks inside 'templates/' folder
    return render_template('index.html')

@app.route('/metrics', methods=['GET'])
def get_metrics():
    _, _, _, metrics = load_ml_resources()
    if metrics is None:
        return jsonify({'error': 'Models have not been trained yet. Please run training pipeline.'}), 404
    return jsonify(metrics)

@app.route('/predict', methods=['POST'])
def predict():
    model, channel_model, scaler, metrics = load_ml_resources()
    
    if model is None or channel_model is None or scaler is None or metrics is None:
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
        
        # Predict optimal campaign channel
        channel_pred_idx = int(channel_model.predict(scaled_input)[0])
        channel_probs = channel_model.predict_proba(scaled_input)[0]
        
        channels = ['Email', 'WhatsApp', 'SMS', 'Phone Call', 'Social Media']
        predicted_channel = channels[channel_pred_idx]
        
        # Create a dictionary of channel probabilities
        channel_prob_dict = {channels[i]: round(float(channel_probs[i]) * 100, 1) for i in range(len(channels))}
        
        # Recommend Personalized Communication Style
        freq = float(data.get('donation_frequency', 1))
        months_since_last = float(data.get('time_since_last_donation', 1))
        
        if freq > 5:
            donor_type = "Frequent Donor"
            rec_message = "Thank-you & impact report"
            details = "Express gratitude for their ongoing support and share a detailed report of the latest community impact achieved through their donations."
        elif freq <= 2 and months_since_last <= 3:
            donor_type = "New Donor"
            rec_message = "Welcome & trust-building"
            details = "Send a warm welcome kit explaining our mission, stories of beneficiaries, and how their initial contribution is being used."
        elif months_since_last > 12:
            donor_type = "Inactive Donor"
            rec_message = "Success stories & reminders"
            details = "Re-engage them by sharing inspiring success stories from recent months along with a gentle invitation to renew their support."
        else:
            donor_type = "Active Donor"
            rec_message = "Personalized updates & appeal"
            details = "Keep them engaged with personalized updates about ongoing campaigns and invite them to participate in current funding drives."
            
        response = {
            'success': True,
            'prediction': 'Yes' if prediction == 1 else 'No',
            'probability': round(prob * 100, 1),
            'model_used': metrics.get('best_model', 'Unknown'),
            
            # New features
            'predicted_channel': predicted_channel,
            'channel_probabilities': channel_prob_dict,
            'channel_model_used': metrics.get('channel_model', {}).get('best_model', 'Unknown'),
            'donor_type': donor_type,
            'recommended_message': rec_message,
            'recommendation_details': details
        }
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting NayePankh Foundation prediction server...")
    # Run server on port 5000
    app.run(debug=True, host='127.0.0.1', port=5000)

