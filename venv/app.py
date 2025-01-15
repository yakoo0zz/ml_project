from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
import re

app = Flask(__name__)
CORS(app)

# Veriyi yükleme
try:
    data = pd.read_csv('data/health_data.csv')
    print("Veriler başarıyla yüklendi.")
except Exception as e:
    print(f"Veriler yüklenirken hata oluştu: {e}")
    data = None

# Değerin aralık içinde olup olmadığını kontrol etme
def is_value_in_range(değer, aralık):
    """
    Girilen değerin aralık içinde olup olmadığını kontrol eder.
    """
    try:
        # '<' işareti için kontrol
        if '<' in aralık:
            limit = float(re.search(r'\d+(\.\d+)?', aralık).group())
            return değer < limit

        # '>' işareti için kontrol
        elif '>' in aralık:
            limit = float(re.search(r'\d+(\.\d+)?', aralık).group())
            return değer > limit

        # '-' ile aralık için kontrol
        elif '-' in aralık:
            lower, upper = map(float, aralık.split('-'))
            return lower <= değer <= upper

        # Format dışı durumlar
        else:
            print(f"Geçersiz aralık formatı: {aralık}")
            return False

    except Exception as e:
        print(f"Aralık kontrolünde hata: {e}")
        return False

@app.route('/predict', methods=['POST'])
def predict():
    """
    Kullanıcıdan gelen parametre ve değere göre öneri döner.
    """
    try:
        # Veriyi kontrol et
        if data is None or data.empty:
            return jsonify({"error": "Sağlık verileri yüklenemedi!"}), 500

        # Kullanıcıdan gelen JSON verisini al
        incoming_data = request.json
        parametre = str(incoming_data.get("parametre", "")).strip()  # Stringe dönüştürme
        değer = incoming_data.get("değer")

        # Değerin sayıya çevrilmesi ve kontrol edilmesi
        try:
            değer = float(değer)
        except (ValueError, TypeError):
            return jsonify({"error": "Geçersiz değer! Lütfen bir sayı girin."}), 400

        if not parametre:
            return jsonify({"error": "Parametre alanı zorunludur!"}), 400

        # İlgili parametreye göre filtrele
        filtered_data = data[data['Parametre'].str.lower() == parametre.lower()]
        if filtered_data.empty:
            return jsonify({"error": f"Girilen parametre bulunamadı: {parametre}"}), 400

        # Değer aralığını kontrol et ve uygun öneriyi bul
        for _, row in filtered_data.iterrows():
            aralık = row['Değer_Aralığı']
            öneri = row['Öneri']

            if is_value_in_range(değer, aralık):
                return jsonify({
                    "parametre": parametre,
                    "değer": değer,
                    "durum": row['Durum'],
                    "öneri": öneri
                }), 200

        return jsonify({
            "parametre": parametre,
            "değer": değer,
            "öneri": "Girilen değer için uygun bir öneri bulunamadı."
        }), 200

    except Exception as e:
        print(f"Bir hata oluştu: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
