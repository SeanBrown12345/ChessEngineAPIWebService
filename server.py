from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load the trained model
try:
    model = joblib.load("chess_rating_optimal_model.pkl")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None



@app.route('/predict', methods=['POST'])
def predict_elo():
    try:
        data = request.json
        time_control = data.get('time_control')
        opening_moves = data.get('opening_moves')
        
        if not time_control or not opening_moves:
            return jsonify({'error': 'Missing required fields'}), 400
        
        input_data = pd.DataFrame({
            'time_control': [time_control],
            'opening_only': [opening_moves]
        })
        
        prediction = model.predict(input_data)[0]
        print(f"Prediction: {prediction}")
        
        return jsonify({
            'predicted_elo': round(float(prediction)),
            'time_control': time_control,
            'opening_moves': opening_moves
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500144.384757387634

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(debug=True, port=5000)