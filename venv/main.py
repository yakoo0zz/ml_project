import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# 1. Veriyi Yükleme
def load_data(file_path):
    """CSV dosyasını yükler ve veri setini döndürür."""
    try:
        data = pd.read_csv(file_path)
        print(f"Veriler başarıyla yüklendi: {file_path}")
        # Gerekli sütunları kontrol et
        required_columns = ['Parametre', 'Değer_Aralığı', 'Durum', 'Öneri']
        for column in required_columns:
            if column not in data.columns:
                raise ValueError(f"CSV dosyasında '{column}' sütunu bulunmalıdır.")
        return data
    except FileNotFoundError:
        print(f"CSV dosyası bulunamadı: {file_path}")
    except ValueError as ve:
        print(f"Veri hatası: {ve}")
    except Exception as e:
        print(f"Beklenmeyen bir hata oluştu: {e}")
    return pd.DataFrame()

# 2. Model Eğitimi
def train_model(data):
    """Modeli eğitir ve kaydeder."""
    try:
        # Soru ve öneri sütunlarını seç
        questions = data['Değer_Aralığı']
        answers = data['Öneri']

        # TF-IDF ile metinleri sayısal verilere dönüştür
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
        X = vectorizer.fit_transform(questions)
        y = answers

        # Veriyi eğitim ve test olarak böl
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Multinomial Naive Bayes Modeli Eğit
        model = MultinomialNB()
        model.fit(X_train, y_train)

        # Test seti doğruluğunu hesapla
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Modeli ve vektörleştiriciyi kaydet
        joblib.dump(model, "model.pkl")
        joblib.dump(vectorizer, "vectorizer.pkl")
        print("Model ve vektörleştirici başarıyla kaydedildi.")
    except Exception as e:
        print(f"Model eğitimi sırasında bir hata oluştu: {e}")

# Ana Akış
if __name__ == "__main__":
    # Veri dosyasının yolu
    file_path = "./data/health_data.csv"

    # Veriyi yükle
    data = load_data(file_path)

    if not data.empty:
        # Modeli eğit
        train_model(data)
    else:
        print("Veri bulunamadı veya hatalı. Model eğitimi yapılamadı.")
