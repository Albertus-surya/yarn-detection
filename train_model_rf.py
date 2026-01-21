import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Load Dataset V2 (Extended)
DATA_FILE = os.path.join(BASE_DIR, "data", "yarn_dataset_extended.csv")
MODEL_FILE = os.path.join(BASE_DIR, "models", "rf_yarn_model_v2.pkl")

def train():
    if not os.path.exists(DATA_FILE):
        print("Dataset extended belum ada. Jalankan collector v2 dulu!")
        return

    df = pd.read_csv(DATA_FILE)
    if df.empty: return

    # --- PENTING: MENGGUNAKAN 6 KOLOM SEBAGAI INPUT ---
    X = df[['L', 'a', 'b', 'std_L', 'std_a', 'std_b']]
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🌲 Training Random Forest dengan 6 Fitur (Mean + Texture)...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"🚀 Akurasi Model V2: {acc*100:.2f}%")
    print(classification_report(y_test, rf.predict(X_test)))

    joblib.dump(rf, MODEL_FILE)
    print(f"💾 Model disimpan ke: {MODEL_FILE}")

if __name__ == "__main__":
    train()