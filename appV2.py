import streamlit as st
import cv2
import numpy as np
import pandas as pd
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase, WebRtcMode
import av
import os
import time
import collections
from datetime import datetime

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Yarn Master System V2 (Stabilized)", layout="wide")

DATABASE_FILE = "database_yarn.csv"
HISTORY_FILE = "riwayat_deteksi.csv"
SNAPSHOT_FOLDER = "snapshots"
ROI_SIZE = 100  # Ukuran area deteksi (lebih kecil sedikit agar fokus)

# Buat folder jika belum ada
if not os.path.exists(SNAPSHOT_FOLDER):
    os.makedirs(SNAPSHOT_FOLDER)

# Init CSV
if not os.path.exists(DATABASE_FILE):
    with open(DATABASE_FILE, 'w') as f:
        f.write("Kode_Warna,L,a,b,R_preview,G_preview,B_preview\n")

if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'w') as f:
        f.write("Waktu,Kode_Warna,Jarak_DeltaE,L,a,b,Mode_Input,File_Gambar\n")

# --- SESSION STATE ---
if "wb_gains" not in st.session_state:
    # Default gains (1.0 artinya tidak ada perubahan warna)
    st.session_state.wb_gains = (1.0, 1.0, 1.0) 

if "confidence_threshold" not in st.session_state:
    st.session_state.confidence_threshold = 10.0  # Diperketat karena DeltaE 2000 lebih presisi

# --- FUNGSI MATEMATIKA & KONVERSI (PENTING) ---

def opencv_lab_to_standard(l_cv, a_cv, b_cv):
    """
    Mengubah format OpenCV (8-bit) ke format Standar CIELAB.
    OpenCV: L(0-255), a(0-255), b(0-255)
    Standard: L(0-100), a(-128..127), b(-128..127)
    """
    L_std = l_cv * 100 / 255.0
    a_std = a_cv - 128
    b_std = b_cv - 128
    return L_std, a_std, b_std

def delta_e_2000(lab1_std, lab2_std):
    """
    Menghitung jarak warna CIEDE2000.
    Input HARUS dalam format Standard (L:0-100, a/b:-128..127)
    """
    L1, a1, b1 = lab1_std
    L2, a2, b2 = lab2_std
    
    kL = kC = kH = 1
    
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_mean = (C1 + C2) / 2
    
    G = 0.5 * (1 - np.sqrt(C_mean**7 / (C_mean**7 + 25**7)))
    a1_prime = a1 * (1 + G)
    a2_prime = a2 * (1 + G)
    
    C1_prime = np.sqrt(a1_prime**2 + b1**2)
    C2_prime = np.sqrt(a2_prime**2 + b2**2)
    
    h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360
    h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360
    
    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime
    
    if C1_prime * C2_prime == 0:
        delta_h_prime = 0
    else:
        diff = h2_prime - h1_prime
        if abs(diff) <= 180:
            delta_h_prime = diff
        elif diff > 180:
            delta_h_prime = diff - 360
        else:
            delta_h_prime = diff + 360
            
    delta_H_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(delta_h_prime / 2))
    
    L_mean = (L1 + L2) / 2
    C_prime_mean = (C1_prime + C2_prime) / 2
    
    if C1_prime * C2_prime == 0:
        h_prime_mean = h1_prime + h2_prime
    else:
        if abs(h1_prime - h2_prime) <= 180:
            h_prime_mean = (h1_prime + h2_prime) / 2
        elif h1_prime + h2_prime < 360:
            h_prime_mean = (h1_prime + h2_prime + 360) / 2
        else:
            h_prime_mean = (h1_prime + h2_prime - 360) / 2
            
    T = 1 - 0.17 * np.cos(np.radians(h_prime_mean - 30)) + \
        0.24 * np.cos(np.radians(2 * h_prime_mean)) + \
        0.32 * np.cos(np.radians(3 * h_prime_mean + 6)) - \
        0.20 * np.cos(np.radians(4 * h_prime_mean - 63))
        
    SL = 1 + (0.015 * (L_mean - 50)**2) / np.sqrt(20 + (L_mean - 50)**2)
    SC = 1 + 0.045 * C_prime_mean
    SH = 1 + 0.015 * C_prime_mean * T
    
    delta_theta = 30 * np.exp(-((h_prime_mean - 275) / 25)**2)
    RC = 2 * np.sqrt(C_prime_mean**7 / (C_prime_mean**7 + 25**7))
    RT = -RC * np.sin(np.radians(2 * delta_theta))
    
    delta_E = np.sqrt(
        (delta_L_prime / (kL * SL))**2 +
        (delta_C_prime / (kC * SC))**2 +
        (delta_H_prime / (kH * SH))**2 +
        RT * (delta_C_prime / (kC * SC)) * (delta_H_prime / (kH * SH))
    )
    return delta_E

# --- FUNGSI UTILITY ---

def apply_manual_wb(img, gains):
    """Menerapkan gain RGB manual untuk kalibrasi"""
    b, g, r = cv2.split(img)
    b = cv2.multiply(b, gains[0]) # Blue Gain
    g = cv2.multiply(g, gains[1]) # Green Gain
    r = cv2.multiply(r, gains[2]) # Red Gain
    merged = cv2.merge([b, g, r])
    return np.clip(merged, 0, 255).astype(np.uint8)

def extract_lab_stable(image_bgr):
    """
    Mengambil sampel warna dari 5 titik (tengah dan sekitarnya)
    Lalu mengambil median untuk menghindari noise.
    """
    h, w, _ = image_bgr.shape
    cx, cy = w // 2, h // 2
    
    # 5 Titik Sampling
    offsets = [(0,0), (-30,-30), (30,-30), (-30,30), (30,30)]
    lab_samples = []
    
    # Blur dulu untuk meratakan tekstur benang
    img_blur = cv2.GaussianBlur(image_bgr, (25, 25), 0)
    img_lab_full = cv2.cvtColor(img_blur, cv2.COLOR_BGR2Lab)
    
    valid_samples = 0
    
    for dx, dy in offsets:
        px, py = cx + dx, cy + dy
        if 0 <= px < w and 0 <= py < h:
            # Ambil area 10x10 pixel
            roi = img_lab_full[py-5:py+5, px-5:px+5]
            if roi.size > 0:
                mean_val = np.mean(roi, axis=(0,1))
                lab_samples.append(mean_val)
                valid_samples += 1
    
    if valid_samples > 0:
        # Ambil Median (lebih tahan outlier daripada Mean)
        final_lab = np.median(lab_samples, axis=0)
        
        # Koordinat kotak visualisasi
        rect = (cx - ROI_SIZE, cy - ROI_SIZE, cx + ROI_SIZE, cy + ROI_SIZE)
        return final_lab, rect
    
    return None, None

def load_database():
    try:
        df = pd.read_csv(DATABASE_FILE)
        df['Kode_Warna'] = df['Kode_Warna'].astype(str)
        return df
    except:
        return pd.DataFrame()

def save_new_data(code, L, a, b, R, G, B):
    try:
        # Cek duplikat nama
        df = load_database()
        if not df.empty and code in df['Kode_Warna'].values:
            code = f"{code}_{int(time.time())}" # Auto rename jika duplikat
            
        with open(DATABASE_FILE, 'a') as f:
            f.write(f"{code},{L},{a},{b},{R},{G},{B}\n")
        return True, code
    except Exception as e:
        return False, str(e)

# --- CLASS PEMROSES VIDEO ---

class YarnDetector(VideoProcessorBase):
    def __init__(self):
        # Buffer untuk smoothing (rata-rata bergerak 10 frame)
        self.l_buffer = collections.deque(maxlen=10)
        self.a_buffer = collections.deque(maxlen=10)
        self.b_buffer = collections.deque(maxlen=10)
        
        self.wb_gains = st.session_state.wb_gains
        self.db_lookup = load_database()
        self.threshold = st.session_state.confidence_threshold
        
        self.latest_result = None # Untuk dikirim ke UI

    def update_settings(self):
        # Dipanggil dari main thread untuk update setting realtime
        self.wb_gains = st.session_state.wb_gains
        self.threshold = st.session_state.confidence_threshold
        # Reload DB sesekali jika perlu, tapi hati-hati performance
        if np.random.rand() > 0.95: 
            self.db_lookup = load_database()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Apply White Balance Manual
        img_wb = apply_manual_wb(img, self.wb_gains)
        
        # 2. Ekstraksi Warna (OpenCV Format)
        lab_cv, rect = extract_lab_stable(img_wb)
        
        result_packet = None
        
        if lab_cv is not None:
            # Masukkan ke buffer smoothing
            self.l_buffer.append(lab_cv[0])
            self.a_buffer.append(lab_cv[1])
            self.b_buffer.append(lab_cv[2])
            
            # Hitung Rata-rata Stabil
            l_smooth = np.mean(self.l_buffer)
            a_smooth = np.mean(self.a_buffer)
            b_smooth = np.mean(self.b_buffer)
            
            # Konversi Rata-rata ke Standard LAB (untuk Delta E)
            lab_std_current = opencv_lab_to_standard(l_smooth, a_smooth, b_smooth)
            
            # Visualisasi Kotak
            if rect:
                cv2.rectangle(img_wb, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
            
            # Pencarian Database
            best_match = None
            min_dist = 999.0
            
            if not self.db_lookup.empty:
                for _, row in self.db_lookup.iterrows():
                    # Konversi Data Database (OpenCV) ke Standard LAB
                    lab_std_db = opencv_lab_to_standard(row['L'], row['a'], row['b'])
                    
                    dist = delta_e_2000(lab_std_current, lab_std_db)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_match = row
            
            # Siapkan hasil untuk UI
            confident = min_dist < self.threshold
            
            kode_res = "Unknown"
            if best_match is not None:
                kode_res = best_match['Kode_Warna']
                color_txt = (0, 255, 0) if confident else (0, 165, 255)
                
                # Tampilkan text di video
                label = f"{kode_res} (dE:{min_dist:.1f})"
                cv2.putText(img_wb, label, (rect[0], rect[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_txt, 2)
            
            result_packet = {
                "kode": kode_res,
                "dist": min_dist,
                "lab_cv": (l_smooth, a_smooth, b_smooth), # Simpan data mentah OpenCV
                "lab_std": lab_std_current,
                "image": img_wb.copy(),
                "confident": confident
            }
            
        self.latest_result = result_packet
        return av.VideoFrame.from_ndarray(img_wb, format="bgr24")

# --- UI UTAMA ---

st.title("🧶 Yarn Master Pro - Precision Color Detector")

# Sidebar
st.sidebar.header("Kontrol Sistem")
menu = st.sidebar.radio("Mode", ["Realtime Detector", "Input Master Data", "Database View"])
st.sidebar.markdown("---")
conf_thresh = st.sidebar.slider("Toleransi Warna (Delta E)", 1.0, 20.0, 10.0)
st.session_state.confidence_threshold = conf_thresh

# MODE 1: REALTIME
if menu == "Realtime Detector":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Kamera")
        rtc_cfg = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        
        ctx = webrtc_streamer(
            key="yarn-detector",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=YarnDetector,
            rtc_configuration=rtc_cfg,
            media_stream_constraints={"video": {"width": 1280, "height": 720}, "audio": False},
            async_processing=True
        )
        
        # Update settings ke processor secara periodik
        if ctx.video_processor:
            ctx.video_processor.update_settings()

    with col2:
        st.subheader("Panel Kontrol")
        
        # --- FITUR KALIBRASI ---
        st.info("💡 **Tips Konsistensi:** Lakukan kalibrasi dengan kertas putih setiap kali cahaya ruangan berubah.")
        if st.button("⚖️ Set White Reference (Kalibrasi)"):
            if ctx.video_processor and ctx.video_processor.latest_result:
                img_sample = ctx.video_processor.latest_result['image']
                
                # Hitung rata-rata tengah gambar
                h, w, _ = img_sample.shape
                roi = img_sample[h//2-50:h//2+50, w//2-50:w//2+50]
                avg_b = np.mean(roi[:,:,0])
                avg_g = np.mean(roi[:,:,1])
                avg_r = np.mean(roi[:,:,2])
                
                # Target brightness (agak terang tapi tidak mentok 255)
                target = 240.0
                
                # Hitung Gain (Pengali)
                gb = target / avg_b if avg_b > 10 else 1.0
                gg = target / avg_g if avg_g > 10 else 1.0
                gr = target / avg_r if avg_r > 10 else 1.0
                
                st.session_state.wb_gains = (gb, gg, gr)
                st.success(f"Terkalibrasi! Gains: B:{gb:.2f} G:{gg:.2f} R:{gr:.2f}")
                time.sleep(1)
                st.rerun()
            else:
                st.warning("Nyalakan kamera dan tunggu gambar muncul dahulu.")
        
        st.markdown("---")
        
        # --- TAMPILAN HASIL ---
        st.write("### Hasil Deteksi")
        if ctx.state.playing and ctx.video_processor:
            res = ctx.video_processor.latest_result
            if res:
                # Indikator Visual
                if res['confident']:
                    st.success(f"MATCH: {res['kode']}")
                else:
                    st.error(f"CLOSEST: {res['kode']} (Tidak Yakin)")
                
                st.metric("Jarak Delta E", f"{res['dist']:.2f}")
                
                # Detail Angka
                with st.expander("Detail Data Lab"):
                    l_c, a_c, b_c = res['lab_cv']
                    l_s, a_s, b_s = res['lab_std']
                    st.write("**Raw OpenCV (0-255):**")
                    st.code(f"L: {l_c:.1f}, a: {a_c:.1f}, b: {b_c:.1f}")
                    st.write("**Standard CIELAB:**")
                    st.code(f"L: {l_s:.1f}, a: {a_s:.1f}, b: {b_s:.1f}")
                
                # Tombol Simpan
                if st.button("📸 Simpan Hasil Deteksi"):
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fname = f"{res['kode']}_{ts}.jpg"
                    fpath = os.path.join(SNAPSHOT_FOLDER, fname)
                    cv2.imwrite(fpath, res['image'])
                    
                    with open(HISTORY_FILE, 'a') as f:
                        f.write(f"{datetime.now()},{res['kode']},{res['dist']:.2f},{l_c:.1f},{a_c:.1f},{b_c:.1f},Realtime,{fname}\n")
                    st.toast("Data tersimpan di History!")

# MODE 2: INPUT MASTER DATA
elif menu == "Input Master Data":
    st.title("Input Data Master Baru")
    
    col_in1, col_in2 = st.columns(2)
    
    with col_in1:
        new_code = st.text_input("Kode Warna Baru", placeholder="Misal: RED-YARN-001")
        st.write("Ambil foto benang untuk dijadikan acuan master.")
        cam_in = st.camera_input("Kamera Master")
    
    with col_in2:
        if cam_in and new_code:
            bytes_data = cam_in.getvalue()
            img_master = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            # Apply Kalibrasi yang sedang aktif agar konsisten
            img_master = apply_manual_wb(img_master, st.session_state.wb_gains)
            
            # Ekstraksi
            lab_cv, rect = extract_lab_stable(img_master)
            
            if lab_cv is not None:
                # Preview crop
                crop = img_master[rect[1]:rect[3], rect[0]:rect[2]]
                st.image(crop, caption="Area Sample", width=150)
                
                # Konversi ke RGB untuk preview di DB
                pixel_lab = np.uint8([[[lab_cv[0], lab_cv[1], lab_cv[2]]]])
                pixel_bgr = cv2.cvtColor(pixel_lab, cv2.COLOR_Lab2BGR)[0][0]
                
                st.write(f"Terbaca (OpenCV): L={lab_cv[0]:.1f}, a={lab_cv[1]:.1f}, b={lab_cv[2]:.1f}")
                
                if st.button("Simpan ke Database", type="primary"):
                    ok, msg = save_new_data(
                        new_code, 
                        lab_cv[0], lab_cv[1], lab_cv[2], # Simpan nilai OpenCV
                        pixel_bgr[2], pixel_bgr[1], pixel_bgr[0] # Simpan RGB Preview
                    )
                    if ok:
                        st.success(f"Berhasil: {msg}")
                        st.balloons()
                    else:
                        st.error(f"Gagal: {msg}")
            else:
                st.error("Gagal mendeteksi warna.")

# MODE 3: DATABASE VIEW
elif menu == "Database View":
    st.title("Database Warna")
    df = load_database()
    
    if not df.empty:
        st.dataframe(df, use_container_width=True)
        
        st.write("### Statistik")
        col_s1, col_s2 = st.columns(2)
        col_s1.metric("Total SKU Warna", len(df))
        col_s1.download_button("Download CSV", df.to_csv(index=False), "yarn_db.csv")
    else:
        st.info("Database kosong.")