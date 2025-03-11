
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
    chroma =