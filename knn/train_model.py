import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# --- 1. KONFIGURASI PATH ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "yarn_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILE = os.path.join(MODEL_DIR, "knn_yarn_model.pkl")

# Buat folder models jika belum ada
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def train_knn_model():
    print(f"📂 Memuat dataset dari: {DATA_FILE}")
    
    # Cek apakah file ada
    if not os.path.exists(DATA_FILE):
        print("❌ Error: File dataset tidak ditemukan! Jalankan data_collector.py dulu.")
        return

    # Load dataset
    df = pd.read_csv(DATA_FILE)
    
    # Cek jika data masih kosong
    if df.empty:
        print("❌ Error: Dataset kosong.")
        return
        
    print(f"✅ Dataset dimuat: {len(df)} baris data.")

    # --- 2. PERSIAPAN DATA (X dan y) ---
    # X (Features) = Data Input (L, a, b)
    X = df[['L', 'a', 'b']]
    
    # y (Target) = Label jawaban (Kode Benang)
    y = df['Label']

    # Bagi data: 80% untuk latihan (Train), 20% untuk ujian (Test)
    # random_state=42 agar hasil pembagiannya konsisten tiap kali dijalankan
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 3. TRAINING MODEL KNN ---
    # n_neighbors=5 artinya dia akan tanya ke 5 tetangga terdekat
    # weights='distance' artinya tetangga yang lebih dekat suaranya lebih didengar
    k_neighbors = 8
    knn = KNeighborsClassifier(n_neighbors=k_neighbors, weights='distance')
    
    print(f"🔄 Sedang melatih model KNN dengan k={k_neighbors}...")
    knn.fit(X_train, y_train)

    # --- 4. EVALUASI MODEL ---
    print("📊 Menguji akurasi model...")
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("-" * 30)
    print(f"🎯 AKURASI MODEL: {accuracy * 100:.2f}%")
    print("-" * 30)
    
    # Tampilkan detail performa per warna
    print("\nLaporan Detail Klasifikasi:")
    print(classification_report(y_test, y_pred))

    # --- 5. SIMPAN MODEL ---
    print(f"💾 Menyimpan model ke: {MODEL_FILE}")
    joblib.dump(knn, MODEL_FILE)
    print("✅ Selesai! Model siap digunakan di aplikasi utama.")

if __name__ == "__main__":
    train_knn_model()