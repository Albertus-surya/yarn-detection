import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import os

# --- KONFIGURASI ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "yarn_dataset.csv")

def analyze():
    # 1. Load Data
    if not os.path.exists(DATA_FILE):
        print("Dataset tidak ditemukan.")
        return
    
    df = pd.read_csv(DATA_FILE)
    X = df[['L', 'a', 'b']]
    y = df['Label']
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("🔍 SEDANG MENCARI NILAI 'K' TERBAIK...")
    
    # 2. Hyperparameter Tuning (Mencari K terbaik)
    best_k = 0
    best_score = 0
    
    results = []
    
    for k in range(1, 21): # Coba K dari 1 sampai 20
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
        # Cross validation biar lebih valid
        scores = cross_val_score(knn, X, y, cv=5)
        avg_score = scores.mean()
        results.append(avg_score)
        
        if avg_score > best_score:
            best_score = avg_score
            best_k = k
            
    print(f"✅ Nilai K Terbaik ditemukan: {best_k} (Akurasi estimasi: {best_score*100:.2f}%)")
    
    # 3. Evaluasi Detail dengan K Terbaik
    final_model = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)
    
    real_accuracy = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Akurasi Final di Test Set: {real_accuracy*100:.2f}%")
    
    # 4. Tampilkan Confusion Matrix (Siapa tertukar dengan siapa)
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(y.unique())
    
    print("\n⚠️ TABEL KEBINGUNGAN (CONFUSION MATRIX):")
    print("Baris = Warna Asli, Kolom = Tebakan AI")
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)
    
    # 5. Visualisasi 3D (Agar kamu paham kenapa akurasi rendah)
    print("\n📊 Membuka Plot 3D...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Map label ke warna untuk plot (hanya visualisasi)
    unique_labels = y.unique()
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        subset = df[df['Label'] == label]
        ax.scatter(subset['L'], subset['a'], subset['b'], label=label, s=20, alpha=0.6)
    
    ax.set_xlabel('L (Lightness)')
    ax.set_ylabel('a (Green-Red)')
    ax.set_zlabel('b (Blue-Yellow)')
    ax.set_title(f'Sebaran Data Warna (Akurasi: {real_accuracy*100:.1f}%)')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    # Instal matplotlib dan seaborn dulu jika belum
    # pip install matplotlib seaborn
    analyze()