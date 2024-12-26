from flask import Flask, request, jsonify
import joblib
import os

# Flask uygulamasını oluştur
app = Flask(__name__)

# Kaydedilen modeli ve vektörleştiriciyi yükle
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "model.pkl")
vectorizer_path = os.path.join(current_dir, "vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Kullanıcıdan gelen soruya cevap tahmini yapar.
    """
    try:
        # İstekten gelen veriyi al
        data = request.json
        question = data.get("question")

        if not question:
            return jsonify({"error": "Question field is required!"}), 400

        # Soru metnini vektörleştir
        question_vector = vectorizer.transform([question])

        # Tahmin yap
        answer = model.predict(question_vector)[0]

        return jsonify({"question": question, "answer": answer}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Flask uygulamasını başlat
if __name__ == "__main__":
    app.run(debug=True)
