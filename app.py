import joblib
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")

app = Flask(__name__)
CORS(app)

city_mapping = {
    "Delhi": 0, "Bengaluru": 1, "Chennai": 2, "Mumbai": 3, "Kolkata": 4,
    "Hyderabad": 5, "Pune": 6, "Ahmedabad": 7, "Jaipur": 8, "Lucknow": 9
}

state_mapping = {
    "Delhi": 0, "Karnataka": 1, "Tamil Nadu": 2, "Maharashtra": 3,
    "West Bengal": 4, "Telangana": 5, "Gujarat": 6, "Rajasthan": 7,
    "Uttar Pradesh": 8
}

type_mapping = {
    "Residential": 0, "Commercial": 1, "Industrial": 2, "Mixed": 3
}

@app.route('/')
def serve_index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        year = int(data['Year'])
        month = int(data['Month'])
        day = int(data['Day'])
        night = int(data['Night'])
        city_encoded = city_mapping.get(data['City'], 0)
        state_encoded = state_mapping.get(data['State'], 0)
        type_encoded = type_mapping.get(data['Type'], 0)

        features = [[year, month, day, night, city_encoded, state_encoded, type_encoded]]
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)

        # The model directly predicts the label ('Low', 'Medium', 'High')
        # We just take the first element from the prediction array.
        predicted_label = prediction[0]

        return jsonify({'prediction': predicted_label})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)