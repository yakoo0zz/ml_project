import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# 1. Veriyi Yükleme
def load_data(file_path):
    """Veriyi CSV dosyasından yükler."""
    try:
        data = pd.read_csv(file_path)
        print(f"Veriler başarıyla yüklendi: {file_path}")
        return data
    except Exception as e:
        print(f"Veri yüklenemedi: {e}")
        return pd.DataFrame()

# 2. Model Eğitimi
def train_model(data):
    """Modeli eğitir ve kaydeder."""
    # Soruları ve cevapları ayır
    questions = data['Question']
    answers = data['Answer']

    # TF-IDF ile metinleri sayısal verilere dönüştür
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(questions)
    y = answers

    # Veriyi eğitim ve test olarak böl
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Logistic Regression Modeli Eğit
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Test seti doğruluğunu hesapla
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Doğruluğu: {accuracy:.2f}")

    # Modeli ve vektörleştiriciyi kaydet
    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    print("Model ve vektörleştirici başarıyla kaydedildi.")

# Ana Akış
if __name__ == "__main__":
    # Veriyi yükle
    data = load_data("./data/combined_questions.csv")

    if not data.empty:
        # Modeli eğit
        train_model(data)
    else:
        print("Veri bulunamadı. Model eğitimi yapılamadı.")
