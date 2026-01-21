import streamlit as st
import cv2
import numpy as np
import pandas as pd
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import av
import os
import time

st.set_page_config(page_title="Collector V2 (Texture)", page_icon="📸", layout="wide")

# --- KONFIGURASI FILE BARU ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SNAPSHOT_DIR = os.path.join(DATA_DIR, "snapshots")
# Nama file CSV kita bedakan agar aman
DATASET_FILE = os.path.join(DATA_DIR, "yarn_dataset_extended.csv")

if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
if not os.path.exists(SNAPSHOT_DIR): os.makedirs(SNAPSHOT_DIR)

# --- INISIALISASI CSV DENGAN 6 FITUR ---
if not os.path.exists(DATASET_FILE):
    # Perhatikan kolom tambahannya: std_L, std_a, std_b
    df_init = pd.DataFrame(columns=["L", "a", "b", "std_L", "std_a", "std_b", "Label"])
    df_init.to_csv(DATASET_FILE, index=False)

def preprocess_image(img):
    # Preprocessing standar
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(img[:, :, 1])
    avg_b = np.average(img[:, :, 2])
    img[:, :, 1] = img[:, :, 1] - ((avg_a - 128) * (img[:, :, 0] / 255.0) * 1.1)
    img[:, :, 2] = img[:, :, 2] - ((avg_b - 128) * (img[:, :, 0] / 255.0) * 1.1)
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    
    # Gray World
    img_float = img.astype(np.float32)
    avg_b, avg_g, avg_r = np.mean(img_float, axis=(0, 1))
    avg_gray = (avg_b + avg_g + avg_r) / 3
    if avg_b == 0: avg_b = 1
    if avg_g == 0: avg_g = 1
    if avg_r == 0: avg_r = 1
    img_float[:, :, 0] = np.clip(img_float[:, :, 0] * (avg_gray / avg_b), 0, 255)
    img_float[:, :, 1] = np.clip(img_float[:, :, 1] * (avg_gray / avg_g), 0, 255)
    img_float[:, :, 2] = np.clip(img_float[:, :, 2] * (avg_gray / avg_r), 0, 255)
    
    return img_float.astype(np.uint8)

class DataCollectorProcessor(VideoProcessorBase):
    def __init__(self):
        self.recording = False
        self.current_label = ""
        self.collected_data = [] 
        self.last_img = None
        self.last_record_time = 0

    def set_recording(self, status, label):
        self.recording = status
        self.current_label = label
        if status: self.collected_data = [] 

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
            # --- EKSTRAKSI 6 FITUR ---
            lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab)
            
            # Fungsi Sakti: Menghitung Mean DAN StdDev sekaligus
            mean, std_dev = cv2.meanStdDev(lab_roi)
            
            L, a, b = mean[0][0], mean[1][0], mean[2][0]
            sL, sa, sb = std_dev[0][0], std_dev[1][0], std_dev[2][0]
            
            if self.recording and self.current_label:
                if time.time() - self.last_record_time > 0.1: # Speed limiter
                    self.collected_data.append({
                        "L": L, "a": a, "b": b,
                        "std_L": sL, "std_a": sa, "std_b": sb, # DATA BARU
                        "Label": self.current_label
                    })
                    self.last_record_time = time.time()
                
                cv2.putText(img, f"REC (6-Feat): {len(self.collected_data)}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # Tampilkan info Std Dev di layar untuk debug
                cv2.putText(img, f"Std: {sL:.1f}, {sa:.1f}, {sb:.1f}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# UI SEDERHANA
st.title("📸 Data Collector V2 (Fitur Tekstur)")
st.caption("Menyimpan Mean + StdDev ke `yarn_dataset_extended.csv`")

col_cam, col_ctrl = st.columns([2, 1])
with col_cam:
    ctx = webrtc_streamer(key="collector-v2", mode=WebRtcMode.SENDRECV,
                         rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
                         video_processor_factory=DataCollectorProcessor,
                         media_stream_constraints={"video": True, "audio": False}, async_processing=True)

with col_ctrl:
    label_input = st.text_input("Label")
    if "rec" not in st.session_state: st.session_state.rec = False
    
    if ctx.video_processor:
        if not st.session_state.rec:
            if st.button("Start Recording"):
                if label_input:
                    st.session_state.rec = True
                    ctx.video_processor.set_recording(True, label_input)
                    st.rerun()
                else: st.error("Isi label!")
        else:
            if st.button("Stop & Save"):
                st.session_state.rec = False
                ctx.video_processor.set_recording(False, "")
                data, _ = ctx.video_processor.get_data()
                if data:
                    pd.DataFrame(data).to_csv(DATASET_FILE, mode='a', header=False, index=False)
                    st.success(f"Tersimpan {len(data)} data!")
                st.rerun()

if os.path.exists(DATASET_FILE):
    df = pd.read_csv(DATASET_FILE)
    st.write(f"Total Data: {len(df)}")
    st.dataframe(df.tail(3))