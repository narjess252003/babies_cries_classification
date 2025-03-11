import os
import sys
# Add project path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
from features.featuresExtraction import extractMfcc, extractChroma, extractSpectral, extractZCR, loading
from preprocessing.dataPreprocessing import preprocess_data

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model (Random Forest)
model = joblib.load(r"C:\Users\INFOKOM\Desktop\stage_pfe\baby_cries_classification\model\random_forest_model.pkl")

# Define class labels
cry_classes = ["belly_pain", "burping", "discomfort", "hungry", "tired"]

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Feature extraction
    y, sr = loading(file_path)  # Load the audio file
    mfcc = extractMfcc(y, sr)
    chroma = extractChroma(y, sr)
    spectral_contrast = extractSpectral(y, sr)
    zcr = extractZCR(y)

    # Combine extracted features
    features = np.hstack([mfcc, chroma, spectral_contrast, zcr])
    features_df = pd.DataFrame(features.reshape(1, -1))  # Ensure correct shape

    # Predict using the trained model
    prediction = model.predict(features_df)[0]  # Get the predicted class index
    predicted_class =prediction  # Convert index to class label
    
    return jsonify({
        "message": "Prediction successful",
        "predicted_class": predicted_class
    })

@app.route("/")
def index():
    return "Babies cries classification"

if __name__ == "__main__":
    app.run(debug=True)
