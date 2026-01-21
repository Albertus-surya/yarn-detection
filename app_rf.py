import streamlit as st
import cv2
import numpy as np
import joblib
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import av
import os
import pandas as pd
from datetime import datetime
from collections import deque, Counter

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Yarn AI Collector", page_icon="🌲", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_yarn_model_v2.pkl")
CSV_FILE = os.path.join(BASE_DIR, "scan_history.csv")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH): return None
    return joblib.load(MODEL_PATH)

rf_model = load_model()

# --- FUNGSI PREPROCESSING ---
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(img[:, :, 1])
    avg_b = np.average(img[:, :, 2])
    img[:, :, 1] = img[:, :, 1] - ((avg_a - 128) * (img[:, :, 0] / 255.0) * 1.1)
    img[:, :, 2] = img[:, :, 2] - ((avg_b - 128) * (img[:, :, 0] / 255.0) * 1.1)
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    img_float = img.astype(np.float32)
    avg_b, avg_g, avg_r = np.mean(img_float, axis=(0, 1))
    avg_gray = (avg_b + avg_g + avg_r) / 3
    if avg_b==0: avg_b=1
    if avg_g==0: avg_g=1
    if avg_r==0: avg_r=1
    img_float[:, :, 0] = np.clip(img_float[:, :, 0] * (avg_gray / avg_b), 0, 255)
    img_float[:, :, 1] = np.clip(img_float[:, :, 1] * (avg_gray / avg_g), 0, 255)
    img_float[:, :, 2] = np.clip(img_float[:, :, 2] * (avg_gray / avg_r), 0, 255)
    return img_float.astype(np.uint8)

# --- DETECTOR & DATA EXTRACTOR ---
class YarnDetectorV2(VideoProcessorBase):
    def __init__(self):
        self.history = deque(maxlen=10)
        self.current_result = None # Variabel penampung data real-time

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        proc_img = preprocess_image(img)
        
        h, w, _ = proc_img.shape
        cx, cy = w//2, h//2
        x1, y1, x2, y2 = cx-50, cy-50, cx+50, cy+50
        
        # Gambar kotak fokus
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        roi = proc_img[y1:y2, x1:x2]
        
        if roi.size > 0 and rf_model:
            lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab)
            
            # Ekstraksi fitur lengkap (6 fitur) sesuai dataset
            mean, std = cv2.meanStdDev(lab_roi)
            
            # Format fitur: [Mean L, Mean a, Mean b, Std L, Std a, Std b]
            features = np.array([[
                mean[0][0], mean[1][0], mean[2][0],
                std[0][0], std[1][0], std[2][0]
            ]])
            
            try:
                # Prediksi
                probs = rf_model.predict_proba(features)
                confidence = np.max(probs)
                pred_label = rf_model.predict(features)[0]
                
                if confidence > 0.6:
                    self.history.append(pred_label)
                    winner = Counter(self.history).most_common(1)[0][0]
                    
                    # Simpan data lengkap ke variabel untuk diambil tombol UI
                    self.current_result = {
                        "label": winner,
                        "confidence": confidence,
                        "mean_l": mean[0][0],
                        "mean_a": mean[1][0],
                        "mean_b": mean[2][0],
                        "std_l": std[0][0],
                        "std_a": std[1][0],
                        "std_b": std[2][0]
                    }

                    # Visualisasi di layar
                    cv2.putText(img, f"{winner}", (x1, y1-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    cv2.putText(img, f"Conf: {confidence:.0%}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                else:
                    self.current_result = None
                    cv2.putText(img, "Unsure...", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    
            except Exception as e: 
                print(f"Error: {e}")
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- USER INTERFACE ---
st.title("🌲 Yarn AI Logger")
st.caption("Scan benang dan simpan langsung ke CSV untuk dataset/riwayat.")

if rf_model:
    # Webrtc Streamer
    ctx = webrtc_streamer(
        key="app-logger", 
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        video_processor_factory=YarnDetectorV2,
        media_stream_constraints={"video": True, "audio": False}, 
        async_processing=True
    )

    # --- LOGIKA PENYIMPANAN ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("") # Spacer

    # Tombol Simpan
    if ctx.video_processor:
        # Tombol dibuat besar agar mudah ditekan
        if st.button("💾 SIMPAN KE CSV", type="primary", use_container_width=True):
            data = ctx.video_processor.current_result
            
            if data:
                # 1. Siapkan data baris baru
                new_row = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "label": data['label'],
                    "confidence": round(data['confidence'], 4),
                    # Fitur Dataset (agar bisa dipakai training ulang)
                    "mean_l": data['mean_l'],
                    "mean_a": data['mean_a'],
                    "mean_b": data['mean_b'],
                    "std_l": data['std_l'],
                    "std_a": data['std_a'],
                    "std_b": data['std_b']
                }
                
                # 2. Convert ke DataFrame
                df_new = pd.DataFrame([new_row])
                
                # 3. Append ke CSV (Header hanya ditulis jika file belum ada)
                try:
                    file_exists = os.path.isfile(CSV_FILE)
                    df_new.to_csv(CSV_FILE, mode='a', header=not file_exists, index=False)
                    
                    st.success(f"✅ Data tersimpan: {data['label']} ({data['confidence']:.1%})")
                    print(f"Saved: {new_row}") # Log ke terminal juga
                except Exception as e:
                    st.error(f"Gagal menyimpan: {e}")
            else:
                st.warning("⚠️ Tidak ada objek terdeteksi atau confidence rendah. Pastikan kotak biru fokus ke benang.")

else:
    st.error("Model 'rf_yarn_model_v2.pkl' tidak ditemukan di folder 'models'.")