import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the vectorizer and model using joblib
tfidf_vectorizer = joblib.load(r"E:\NLP(Sentimate analysis project)\tfidf_vectorizer.pkl")
model = joblib.load(r"E:\NLP(Sentimate analysis project)\random_forest_model.pkl")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask model API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json.get("text")  # Get the input text
        if not data:
            return jsonify({"error": "No input text provided"}), 400
        
        # Transform input text using TF-IDF vectorizer
        transformed_text = tfidf_vectorizer.transform([data])

        # Predict using the Random Forest model
        prediction = model.predict(transformed_text)
        
        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
