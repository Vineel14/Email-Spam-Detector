from flask import Flask, request, jsonify
import joblib
import json
import numpy as np
from feature_extractor import analyze_spam_text, scale_vector
from flask_cors import CORS
import os

# Initialize Flask app
app = Flask(__name__)

# Enable CORS
CORS(app)

# Define base directory for relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths to saved models
MODEL_DIR = os.path.join(BASE_DIR, "../models")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.joblib")
NN_MODEL_PATH = os.path.join(MODEL_DIR, "nn_model.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.joblib")

# Load the models
print("Loading models...")
rf_model = joblib.load(RF_MODEL_PATH)  # Random Forest model
nn_model = joblib.load(NN_MODEL_PATH)  # Neural Network model
encoder = joblib.load(ENCODER_PATH)   # OneHotEncoder
print("Models loaded successfully.")

# Load feature min and max values for scaling
CONFIG_PATH = os.path.join(BASE_DIR, "data/feature_min_max.json")
with open(CONFIG_PATH, "r") as f:
    min_max_values = json.load(f)

feature_min = np.array(min_max_values["min"])
feature_max = np.array(min_max_values["max"])

# Load spam keywords and regex patterns
SPAMBASE_NAMES_PATH = os.path.join(BASE_DIR, "data/spambase.names")
with open(SPAMBASE_NAMES_PATH, 'r') as file:
    spambase_names = file.readlines()

spam_words = [
    line.split(":")[0].replace("word_freq_", "").strip()
    for line in spambase_names
    if line.startswith("word_freq_") and "word_freq_table" not in line
]

spam_chars = [
    line.split(":")[0].replace("char_freq_", "").strip()
    for line in spambase_names
    if line.startswith("char_freq_")
]

# Define the predict route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse input text from request
        data = request.get_json()
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "No input text provided"}), 400

        # Step 1: Extract features from the input text
        spam_indicators = analyze_spam_text(text, spam_words, spam_chars)
        input_vector = list(spam_indicators.values())

        # Step 2: Scale the input vector
        scaled_vector = scale_vector(input_vector, feature_min, feature_max)

        # Debugging: Print scaled features
        print("\nScaled Features:")
        print(scaled_vector)

        # Step 3: Get leaf indices from Random Forest
        rf_leaf_indices = rf_model.apply([scaled_vector])  # Pass through RF to get leaf indices

        # Debugging: Print Random Forest leaf indices
        print("\nRandom Forest Leaf Indices:")
        print(rf_leaf_indices)

        # Step 4: One-hot encode the leaf indices
        leaf_encoded_vector = encoder.transform(rf_leaf_indices).toarray()

        # Debugging: Print One-Hot Encoded Features
        print("\nOne-Hot Encoded Features:")
        print(leaf_encoded_vector)

        # Step 5: Predict using the Neural Network
        nn_probabilities = nn_model.predict_proba(leaf_encoded_vector)[0]  # Get probabilities for all classes
        print("\nNeural Network Probabilities:")
        print(nn_probabilities)

        # Final prediction
        spam_probability = nn_probabilities[1]  # Probability of spam class (1)
        prediction = 1 if spam_probability > 0.2 else 0

        # Step 6: Return the prediction result
        result = {
            "input_text": text,
            "prediction": "spam" if prediction == 1 else "not spam",
            "probabilities": list(nn_probabilities)  # Include spam and not spam probabilities
        }
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
