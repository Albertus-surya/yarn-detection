import streamlit as st
import cv2
import numpy as np
import joblib
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import av
import os
from collections import deque, Counter

# Set page config harus di baris paling atas
st.set_page_config(page_title="Yarn Master AI", page_icon="🧶", layout="wide")

# --- 1. CONFIG & LOAD MODEL ---
# Menggunakan os.path.join agar aman di semua OS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "knn_yarn_model.pkl")

# Load Model dengan Caching agar tidak di-load berulang kali setiap frame
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    print(f"Loading model from: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

knn_model = load_model()

# --- 2. PREPROCESSING (WAJIB SAMA DENGAN TRAIN/DATA COLLECTOR) ---
def apply_white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

def gray_world_normalization(img):
    img_float = img.astype(np.float32)
    avg_b, avg_g, avg_r = np.mean(img_float, axis=(0, 1))
    avg_gray = (avg_b + avg_g + avg_r) / 3
    scale_b = avg_gray / avg_b if avg_b != 0 else 1
    scale_g = avg_gray / avg_g if avg_g != 0 else 1
    scale_r = avg_gray / avg_r if avg_r != 0 else 1
    img_float[:, :, 0] = np.clip(img_float[:, :, 0] * scale_b, 0, 255)
    img_float[:, :, 1] = np.clip(img_float[:, :, 1] * scale_g, 0, 255)
    img_float[:, :, 2] = np.clip(img_float[:, :, 2] * scale_r, 0, 255)
    return img_float.astype(np.uint8)

def preprocess_image(img):
    img = gray_world_normalization(img)
    img = apply_white_balance(img)
    img = cv2.GaussianBlur(img, (3,3), 0)
    return img

# --- 3. CORE DETECTOR CLASS ---
class YarnDetector(VideoProcessorBase):
    def __init__(self):
        # STABILIZER: Menyimpan 15 hasil prediksi terakhir
        # Deque otomatis membuang data lama jika sudah penuh
        self.history = deque(maxlen=15)
        self.last_prediction = "Mendeteksi..."
        self.confidence_display = 0.0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Preprocessing
        img_processed = preprocess_image(img)
        
        # 2. ROI (Kotak Tengah)
        h, w, _ = img_processed.shape
        cx, cy = w // 2, h // 2
        roi_size = 50
        x1, y1, x2, y2 = cx - roi_size, cy - roi_size, cx + roi_size, cy + roi_size
        
        # Visualisasi Kotak Scan (Kuning)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # 3. Prediksi
        roi = img_processed[y1:y2, x1:x2]
        
        # Pastikan model ada dan ROI tidak kosong
        if roi.size > 0 and knn_model is not None:
            # Ambil rata-rata LAB
            avg_lab = np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2Lab), axis=(0, 1))
            
            # Format input harus array 2D: [[L, a, b]]
            features = np.array([[int(avg_lab[0]), int(avg_lab[1]), int(avg_lab[2])]])
            
            try:
                # Prediksi langsung (cepat tapi mungkin jittery)
                raw_pred = knn_model.predict(features)[0]
                
                # Masukkan ke history untuk voting (Stabilizer)
                self.history.append(raw_pred)
                
                # --- LOGIKA VOTING ---
                # Cari label yang paling sering muncul di 15 frame terakhir
                counter = Counter(self.history)
                most_common = counter.most_common(1) # Ambil juara 1
                
                if most_common:
                    winner_label = most_common[0][0]
                    winner_count = most_common[0][1]
                    
                    # Hitung 'Kekuatan' kemenangan (Confidence visual)
                    # Jika 15/15 frame bilang "Merah", score = 1.0 (Sangat Yakin)
                    stability_score = winner_count / len(self.history)
                    
                    # Update tampilan hanya jika cukup stabil (> 50% history setuju)
                    if stability_score > 0.5:
                        self.last_prediction = winner_label
                        self.confidence_display = stability_score
                        color_status = (0, 255, 0) # Hijau (Yakin)
                    else:
                        color_status = (0, 165, 255) # Orange (Ragu)
                    
                    # TAMPILAN KE LAYAR
                    text = f"BENANG: {self.last_prediction}"
                    conf_text = f"Stabil: {int(stability_score * 100)}%"
                    
                    # Background Hitam untuk teks agar terbaca jelas
                    cv2.rectangle(img, (x1-10, y1-65), (x2+100, y1-5), (0,0,0), -1)
                    
                    # Tulis Teks
                    cv2.putText(img, text, (x1, y1-35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_status, 2)
                    cv2.putText(img, conf_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Debug Info (L A B realtime di pojok bawah)
                cv2.putText(img, f"L:{int(avg_lab[0])} a:{int(avg_lab[1])} b:{int(avg_lab[2])}", 
                           (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                           
            except Exception as e:
                print(f"Error prediksi: {e}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 4. UI FRONTEND ---
col_logo, col_title = st.columns([1, 5])
with col_title:
    st.title("Yarn Master AI Detection")
    st.write("Sistem deteksi benang berbasis Machine Learning (KNN)")

if knn_model is None:
    st.error(f"❌ Model tidak ditemukan di path: {MODEL_PATH}")
    st.warning("⚠️ Jalankan 'train_model.py' terlebih dahulu untuk membuat file model.")
else:
    col_main, col_info = st.columns([3, 1])
    
    with col_main:
        # Menjalankan Streamer
        ctx = webrtc_streamer(
            key="yarn-detection",
            mode=WebRtcMode.SENDRECV, # Menggunakan Enum WebRtcMode
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            video_processor_factory=YarnDetector,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )
    
    with col_info:
        st.info("ℹ️ **Petunjuk**")
        st.markdown("""
        1. Arahkan kotak kuning ke benang.
        2. Tunggu indikator **Hijau**.
        3. Jika **Orange**, AI sedang bingung (menstabilkan).
        """)
        
        st.markdown("---")
        st.write("🔍 **Diagnosa Model**")
        try:
            classes = knn_model.classes_
            st.write(f"Mengenali {len(classes)} Warna:")
            st.code("\n".join(classes))
        except:
            pass