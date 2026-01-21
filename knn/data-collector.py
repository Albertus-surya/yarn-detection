import streamlit as st
import cv2
import numpy as np
import pandas as pd
# UPDATE IMPORT: Menambahkan WebRtcMode
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import av
import os
import time

st.set_page_config(page_title="Yarn Data Collector", page_icon="📸", layout="wide")

# --- 1. KONFIGURASI STRUKTUR FOLDER ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SNAPSHOT_DIR = os.path.join(DATA_DIR, "snapshots")
DATASET_FILE = os.path.join(DATA_DIR, "yarn_dataset.csv")

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(SNAPSHOT_DIR):
    os.makedirs(SNAPSHOT_DIR)

if not os.path.exists(DATASET_FILE):
    df_init = pd.DataFrame(columns=["L", "a", "b", "Label"])
    df_init.to_csv(DATASET_FILE, index=False)

# --- 2. FUNGSI PREPROCESSING ---
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

# --- 3. CLASS PEMROSES VIDEO ---
class DataCollectorProcessor(VideoProcessorBase):
    def __init__(self):
        self.recording = False
        self.current_label = ""
        self.collected_data = [] 
        self.last_img = None
        self.last_record_time = 0
        self.record_interval = 0.1 

    def set_recording(self, status, label):
        self.recording = status
        self.current_label = label
        if status:
            self.collected_data = [] 

    def get_data(self):
        return self.collected_data, self.last_img

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_processed = preprocess_image(img)
        self.last_img = img.copy() 
        
        h, w, _ = img_processed.shape
        cx, cy = w // 2, h // 2
        roi_size = 50 
        x1, y1, x2, y2 = cx - roi_size, cy - roi_size, cx + roi_size, cy + roi_size
        
        color_box = (0, 0, 255) if self.recording else (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color_box, 2)
        
        roi = img_processed[y1:y2, x1:x2]
        if roi.size > 0:
            avg_lab = np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2Lab), axis=(0, 1))
            L, a, b = int(avg_lab[0]), int(avg_lab[1]), int(avg_lab[2])
            
            current_time = time.time()
            if self.recording and self.current_label:
                if current_time - self.last_record_time > self.record_interval:
                    self.collected_data.append({
                        "L": L, "a": a, "b": b, "Label": self.current_label
                    })
                    self.last_record_time = current_time
                
                cv2.putText(img, f"REC: {len(self.collected_data)} samples", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(img, f"Target: {self.current_label}", (50, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(img, f"L:{L} a:{a} b:{b}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 4. UI ---
st.title("📸 Yarn Dataset Collector")
st.caption(f"Menyimpan data ke: {DATASET_FILE}")

col_cam, col_ctrl = st.columns([2, 1])

with col_cam:
    ctx = webrtc_streamer(
        key="data-collector",
        # FIX: Menggunakan Enum WebRtcMode
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        video_processor_factory=DataCollectorProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

with col_ctrl:
    st.subheader("Kontrol Input")
    label_input = st.text_input("Kode Warna (Label)", placeholder="Contoh: MERAH-001")
    
    if "is_recording" not in st.session_state:
        st.session_state.is_recording = False

    if ctx.video_processor:
        if not st.session_state.is_recording:
            btn_start = st.button("🔴 Mulai Rekam", type="primary", use_container_width=True)
            if btn_start:
                if not label_input:
                    st.error("⚠️ Masukkan Label dulu!")
                else:
                    st.session_state.is_recording = True
                    ctx.video_processor.set_recording(True, label_input)
                    st.rerun()
        else:
            st.info("Sedang merekam... Gerakkan benang sedikit!")
            btn_stop = st.button("💾 Stop & Simpan", type="secondary", use_container_width=True)
            if btn_stop:
                st.session_state.is_recording = False
                ctx.video_processor.set_recording(False, "")
                new_data, last_image = ctx.video_processor.get_data()
                
                if new_data:
                    df_new = pd.DataFrame(new_data)
                    df_new.to_csv(DATASET_FILE, mode='a', header=False, index=False)
                    if last_image is not None:
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        filename = f"{label_input}_{timestamp}.jpg"
                        filepath = os.path.join(SNAPSHOT_DIR, filename)
                        cv2.imwrite(filepath, last_image)
                    st.success(f"✅ Berhasil menyimpan {len(new_data)} data!")
                else:
                    st.warning("Data kosong.")
                st.rerun()

st.divider()
st.subheader("📊 Isi Dataset Saat Ini")
if os.path.exists(DATASET_FILE):
    df = pd.read_csv(DATASET_FILE)
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Total Sampel", len(df))
    with c2: st.metric("Jumlah Warna", df['Label'].nunique())
    
    if not df.empty:
        st.dataframe(df['Label'].value_counts(), use_container_width=True)
        if st.button("⚠️ Hapus Semua Dataset (Reset)", type="primary"):
            df_kosong = pd.DataFrame(columns=["L", "a", "b", "Label"])
            df_kosong.to_csv(DATASET_FILE, index=False)
            st.warning("Reset!")
            st.rerun()