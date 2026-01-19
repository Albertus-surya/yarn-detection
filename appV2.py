import streamlit as st
import cv2
import numpy as np
import pandas as pd
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase, WebRtcMode
import av
import os
import time
from datetime import datetime

st.set_page_config(page_title="Yarn Master System", layout="wide")

DATABASE_FILE = "database_yarn.csv"
HISTORY_FILE = "riwayat_deteksi.csv"
SNAPSHOT_FOLDER = "snapshots"
ROI_SIZE = 140

if not os.path.exists(SNAPSHOT_FOLDER):
    os.makedirs(SNAPSHOT_FOLDER)

# Inisialisasi File CSV
if not os.path.exists(DATABASE_FILE):
    with open(DATABASE_FILE, 'w') as f:
        f.write("Kode_Warna,L,a,b,R_preview,G_preview,B_preview\n")

if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'w') as f:
        f.write("Waktu,Kode_Warna,Jarak_DeltaE,L,a,b,Mode_Input,File_Gambar\n")

# Session state default
if "use_white_balance" not in st.session_state:
    st.session_state.use_white_balance = True
if "use_delta_e_2000" not in st.session_state:
    st.session_state.use_delta_e_2000 = True
if "use_multiple_roi" not in st.session_state:
    st.session_state.use_multiple_roi = True
if "confidence_threshold" not in st.session_state:
    st.session_state.confidence_threshold = 15.0

# --- FUNGSI PREPROCESSING ---

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

def preprocess_image(img, use_wb=True):
    if use_wb:
        img = gray_world_normalization(img)
        img = apply_white_balance(img)
    return img

# --- FUNGSI EKSTRAKSI WARNA ---

def extract_color_roi_filtered(image_bgr):
    h, w, _ = image_bgr.shape
    cx, cy = w // 2, h // 2
    x1, y1, x2, y2 = cx - ROI_SIZE, cy - ROI_SIZE, cx + ROI_SIZE, cy + ROI_SIZE
    
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h: 
        return None, None, None

    roi = image_bgr[y1:y2, x1:x2]
    
    pixels = roi.reshape(-1, 3).astype(np.float32)
    mean = np.mean(pixels, axis=0)
    std = np.std(pixels, axis=0)
    
    mask = np.all(np.abs(pixels - mean) < 2 * std, axis=1)
    filtered_pixels = pixels[mask]
    
    if len(filtered_pixels) > 0:
        avg_bgr = np.mean(filtered_pixels, axis=0)
    else:
        avg_bgr = mean
    
    pixel_bgr = np.uint8([[avg_bgr]])
    pixel_lab = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2Lab)[0][0]
    
    L, a, b = int(pixel_lab[0]), int(pixel_lab[1]), int(pixel_lab[2])
    return (L, a, b), avg_bgr, (x1, y1, x2, y2)

def extract_color_multiple_roi(image_bgr):
    h, w, _ = image_bgr.shape
    cx, cy = w // 2, h // 2
    
    points = [
        (cx, cy),
        (cx-30, cy-30),
        (cx+30, cy-30),
        (cx-30, cy+30),
        (cx+30, cy+30),
    ]
    
    lab_samples = []
    sample_size = 20
    
    for px, py in points:
        if sample_size < px < w-sample_size and sample_size < py < h-sample_size:
            roi = image_bgr[py-sample_size:py+sample_size, px-sample_size:px+sample_size]
            avg_bgr = np.mean(roi, axis=(0, 1))
            pixel_bgr = np.uint8([[avg_bgr]])
            pixel_lab = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2Lab)[0][0]
            lab_samples.append(pixel_lab)
    
    if lab_samples:
        lab_median = np.median(lab_samples, axis=0)
        
        x1, y1 = cx - ROI_SIZE, cy - ROI_SIZE
        x2, y2 = cx + ROI_SIZE, cy + ROI_SIZE
        
        avg_bgr = np.mean([image_bgr[py-10:py+10, px-10:px+10].mean(axis=(0,1)) 
                          for px, py in points if sample_size < px < w-sample_size and sample_size < py < h-sample_size], axis=0)
        
        return tuple(lab_median.astype(int)), avg_bgr, (x1, y1, x2, y2)
    
    return None, None, None

# --- FUNGSI DELTA E ---

def delta_e_2000(lab1, lab2):
    # Implementasi standar CIEDE2000
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
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

def delta_e_simple(lab1, lab2):
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    return np.sqrt((L2-L1)**2 + (a2-a1)**2 + (b2-b1)**2)

# --- FUNGSI MATCHING ---

def find_closest_with_confidence(detected_lab, db_df, use_delta_e_2000=True, threshold=15.0):
    if db_df.empty: 
        return None, 0, False, "Database kosong"
    
    delta_func = delta_e_2000 if use_delta_e_2000 else delta_e_simple
    
    distances = db_df.apply(
        lambda row: delta_func(detected_lab, (row['L'], row['a'], row['b'])), 
        axis=1
    )
    
    min_idx = distances.idxmin()
    min_distance = distances[min_idx]
    
    confident = min_distance < threshold
    warning_msg = ""
    
    if min_distance >= threshold:
        warning_msg = f"Jarak terlalu besar ({min_distance:.1f})"
    
    sorted_dist = distances.nsmallest(2)
    if len(sorted_dist) >= 2:
        gap = sorted_dist.iloc[1] - sorted_dist.iloc[0]
        if gap < 3:
            confident = False
            second_match = db_df.loc[sorted_dist.index[1], 'Kode_Warna']
            warning_msg = f"Ambigu dengan {second_match} (gap: {gap:.1f})"
    
    return db_df.loc[min_idx], min_distance, confident, warning_msg

# --- UTILITIES ---

def load_database(csv_path):
    try:
        df = pd.read_csv(csv_path)
        df['Kode_Warna'] = df['Kode_Warna'].astype(str)
        cols = ['L', 'a', 'b', 'R_preview', 'G_preview', 'B_preview']
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except:
        return pd.DataFrame()

def save_snapshot(image_bgr, kode):
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{kode}_{timestamp_str}.jpg"
    filepath = os.path.join(SNAPSHOT_FOLDER, filename)
    cv2.imwrite(filepath, image_bgr)
    return filename

def save_to_history(kode, distance, lab_values, source_mode, image_filename="-"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(HISTORY_FILE, mode='a', newline='\n') as f:
        row = f"{timestamp},{kode},{distance:.2f},{lab_values[0]},{lab_values[1]},{lab_values[2]},{source_mode},{image_filename}\n"
        f.write(row)

def save_new_master_data(code, l, a, b, r, g, b_val):
    try:
        # Cek apakah kode sudah ada, jika ya buat variasi dengan angka
        curr_db = pd.read_csv(DATABASE_FILE)
        existing_codes = curr_db['Kode_Warna'].values.astype(str).tolist()
        
        original_code = code
        counter = 1
        
        # Jika kode sudah ada, tambahkan suffix (1), (2), dst
        while code in existing_codes:
            code = f"{original_code}({counter})"
            counter += 1
        
        with open(DATABASE_FILE, mode='a', newline='\n') as f:
            f.write(f"{code},{l},{a},{b},{r},{g},{b_val}\n")
        
        if code != original_code:
            return True, f"Berhasil disimpan sebagai '{code}' (nama asli sudah ada)."
        else:
            return True, "Berhasil disimpan."
    except Exception as e:
        return False, str(e)

def lab_to_bgr_preview(l, a, b):
    l_scaled = np.clip(l * 2.55, 0, 255)
    a_scaled = np.clip(a + 128, 0, 255)
    b_scaled = np.clip(b + 128, 0, 255)
    pixel_lab = np.uint8([[[l_scaled, a_scaled, b_scaled]]]) 
    pixel_bgr = cv2.cvtColor(pixel_lab, cv2.COLOR_Lab2BGR)[0][0]
    return int(pixel_bgr[2]), int(pixel_bgr[1]), int(pixel_bgr[0])

# --- VIDEO PROCESSOR CLASS  ---

class YarnDetector(VideoProcessorBase):
    def __init__(self):
        # Default settings
        self.use_wb = True
        self.use_multi = True
        self.use_de2k = True
        self.threshold = 15.0
        
        # Database cache
        try:
            self.db_lookup = pd.read_csv(DATABASE_FILE)
        except:
            self.db_lookup = pd.DataFrame()
            
        # Result container (Thread-safe assignment)
        self.latest_result = None

    def update_settings(self, wb, multi, de2k, thresh):
        self.use_wb = wb
        self.use_multi = multi
        self.use_de2k = de2k
        self.threshold = thresh

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Preprocessing
        img_processed = preprocess_image(img, use_wb=self.use_wb)
        
        # Ekstraksi
        if self.use_multi:
            lab_vals, _, coords = extract_color_multiple_roi(img_processed)
        else:
            lab_vals, _, coords = extract_color_roi_filtered(img_processed)
        
        result_packet = None
        
        if lab_vals and not self.db_lookup.empty:
            x1, y1, x2, y2 = coords
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Pencarian
            best_match, dist, confident, warning = find_closest_with_confidence(
                lab_vals, 
                self.db_lookup, 
                use_delta_e_2000=self.use_de2k,
                threshold=self.threshold
            )
            
            if best_match is not None:
                kode = str(best_match['Kode_Warna'])
                color_text = (0, 255, 0) if confident else (0, 165, 255)
                text = f"{kode} ({dist:.1f})"
                cv2.putText(img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_text, 2)
                
                # Simpan hasil untuk diakses UI
                result_packet = {
                    "kode": kode, "dist": dist, "lab": lab_vals,
                    "image": img.copy(), "confident": confident, "warning": warning
                }

        self.latest_result = result_packet
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI ---

yarn_db = load_database(DATABASE_FILE)

st.sidebar.title("Pengaturan & Navigasi")

# Settings
with st.sidebar.expander("Pengaturan Deteksi", expanded=False):
    st.session_state.use_white_balance = st.checkbox(
        "White Balance Auto", 
        value=st.session_state.use_white_balance
    )
    st.session_state.use_delta_e_2000 = st.checkbox(
        "Delta E 2000", 
        value=st.session_state.use_delta_e_2000
    )
    st.session_state.use_multiple_roi = st.checkbox(
        "Multiple Sampling", 
        value=st.session_state.use_multiple_roi
    )
    st.session_state.confidence_threshold = st.slider(
        "Threshold Confidence",
        min_value=5.0, max_value=30.0, value=st.session_state.confidence_threshold, step=1.0
    )

st.sidebar.markdown("---")

menu = st.sidebar.radio("Pilih Halaman", [
    "Deteksi Realtime", 
    "Deteksi Upload",
    "Database Manager"
])
st.sidebar.markdown("---")

with st.sidebar:
    st.caption("Status Sistem")
    st.caption(f"White Balance: {'ON' if st.session_state.use_white_balance else 'OFF'}")
    st.caption(f"Delta E 2000: {'ON' if st.session_state.use_delta_e_2000 else 'OFF'}")

# DETEKSI REALTIME 
if menu == "Deteksi Realtime":
    st.title("Deteksi Realtime")
    st.write("Arahkan kamera ke objek.")
    
    col_cam, col_ctrl = st.columns([3, 1])
    
    with col_cam:
        rtc_cfg = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        
        # Inisialisasi streamer dengan Processor Class
        ctx = webrtc_streamer(
            key="realtime-view",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=YarnDetector,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 1280},
                    "height": {"ideal": 720},
                    "frameRate": {"ideal": 30}
                },
                "audio": False
            },
            async_processing=True
        )

        # Update settings ke processor jika kamera aktif
        if ctx.video_processor:
            ctx.video_processor.update_settings(
                st.session_state.use_white_balance,
                st.session_state.use_multiple_roi,
                st.session_state.use_delta_e_2000,
                st.session_state.confidence_threshold
            )

    with col_ctrl:
        st.subheader("Kontrol")
        
        if st.button("Simpan Data & Gambar", type="primary", use_container_width=True):
            if ctx.video_processor and ctx.video_processor.latest_result:
                try:
                    data = ctx.video_processor.latest_result
                    
                    kode = data['kode']
                    dist = data['dist']
                    lab = data['lab']
                    img_capture = data['image']
                    confident = data['confident']
                    warning = data['warning']
                    
                    filename_img = save_snapshot(img_capture, kode)
                    save_to_history(kode, dist, lab, "Realtime", filename_img)
                    
                    if confident:
                        st.success(f"Tersimpan: {kode} (Confidence: HIGH)")
                    else:
                        st.warning(f"Tersimpan: {kode} (Confidence: LOW)\n{warning}")
                    
                    st.caption(f"File: {filename_img}")
                    st.caption(f"LAB: {lab[0]}, {lab[1]}, {lab[2]}")
                    
                except Exception as e:
                    st.error(f"Gagal: {e}")
            else:
                st.warning("Kamera belum aktif atau objek belum terdeteksi.")

        st.markdown("---")
        st.write("Riwayat Terakhir:")
        if os.path.exists(HISTORY_FILE):
            try:
                df_hist = pd.read_csv(HISTORY_FILE)
                st.dataframe(
                    df_hist[['Waktu', 'Kode_Warna', 'Jarak_DeltaE']].tail(5), 
                    hide_index=True,
                    use_container_width=True
                )
            except: 
                st.caption("Belum ada riwayat")

# DETEKSI UPLOAD
elif menu == "Deteksi Upload":
    st.title("Deteksi Upload")
    
    uploaded_file = st.file_uploader("Upload gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        col_img, col_res = st.columns(2)
        
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_original = cv2.imdecode(file_bytes, 1)
        
        # Preprocessing
        img_processed = preprocess_image(img_original, use_wb=st.session_state.use_white_balance)
        
        # Ekstraksi
        if st.session_state.use_multiple_roi:
            lab_res, _, coords = extract_color_multiple_roi(img_processed)
        else:
            lab_res, _, coords = extract_color_roi_filtered(img_processed)
        
        with col_img:
            st.subheader("Gambar Input")
            
            img_display = img_original.copy()
            if coords:
                cv2.rectangle(img_display, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 3)
                if st.session_state.use_multiple_roi:
                    h, w, _ = img_display.shape
                    cx, cy = w // 2, h // 2
                    points = [(cx, cy), (cx-30, cy-30), (cx+30, cy-30), (cx-30, cy+30), (cx+30, cy+30)]
                    for px, py in points:
                        cv2.circle(img_display, (px, py), 5, (0, 255, 255), -1)
            
            st.image(img_display, channels="BGR", use_column_width=True)
            
            with st.expander("Lihat Hasil Preprocessing"):
                st.image(img_processed, channels="BGR", use_column_width=True, caption="Gambar setelah Processing")
            
        with col_res:
            st.subheader("Hasil Deteksi")
            
            if lab_res:
                match_row, dist, confident, warning = find_closest_with_confidence(
                    lab_res, 
                    yarn_db,
                    use_delta_e_2000=st.session_state.use_delta_e_2000,
                    threshold=st.session_state.confidence_threshold
                )
                
                if match_row is not None:
                    if confident:
                        st.success(f"Terdeteksi: {match_row['Kode_Warna']}")
                        st.caption("Confidence: HIGH")
                    else:
                        st.warning(f"Terdeteksi: {match_row['Kode_Warna']}")
                        st.caption("Confidence: LOW")
                        if warning:
                            st.caption(warning)
                    
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.metric("Delta E", f"{dist:.2f}")
                    with col_m2:
                        method = "ΔE 2000" if st.session_state.use_delta_e_2000 else "ΔE Simple"
                        st.metric("Metode", method)
                    
                    st.code(f"L: {lab_res[0]}\na: {lab_res[1]}\nb: {lab_res[2]}")
                    
                    st.caption("Warna Database:")
                    color_preview_html = f"""
                    <div style="width:100%; height:50px; background-color:rgb({int(match_row['R_preview'])}, {int(match_row['G_preview'])}, {int(match_row['B_preview'])}); border:2px solid #ccc; border-radius:5px;"></div>
                    """
                    st.markdown(color_preview_html, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    if st.button("Simpan Hasil", type="primary", use_container_width=True):
                        filename_img = save_snapshot(img_original, match_row['Kode_Warna'])
                        save_to_history(match_row['Kode_Warna'], dist, lab_res, "Upload", filename_img)
                        st.success("Data berhasil tersimpan!")
                else:
                    st.error("Tidak ada warna yang cocok di database")
            else:
                st.error("Gagal mengekstrak warna dari gambar")

# DATABASE MANAGER
elif menu == "Database Manager":
    st.title("Database Manager")
    
    tab1, tab2, tab3 = st.tabs(["Lihat Data", "Tambah Data", "Analisis"])
    
    # TAB 1: Lihat Data
    with tab1:
        st.subheader("Database Warna Master")
        
        if not yarn_db.empty:
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.metric("Total Warna", len(yarn_db))
            with col_s2:
                st.metric("L Range", f"{yarn_db['L'].min():.0f} - {yarn_db['L'].max():.0f}")
            with col_s3:
                avg_chroma = np.sqrt(yarn_db['a']**2 + yarn_db['b']**2).mean()
                st.metric("Avg Chroma", f"{avg_chroma:.1f}")
            
            st.markdown("---")
            
            st.dataframe(
                yarn_db.style.background_gradient(subset=['L'], cmap='gray'),
                use_container_width=True, 
                height=400
            )
            
            if st.button("Download Database (CSV)"):
                csv = yarn_db.to_csv(index=False)
                st.download_button(
                    label="Klik untuk Download",
                    data=csv,
                    file_name=f"yarn_database_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("Database kosong. Tambahkan data di tab 'Tambah Data'")

    # TAB 2: Tambah Data
    with tab2:
        st.subheader("Tambah Data Master Baru")
        
        input_type = st.radio(
            "Metode Input", 
            ["Ambil Foto (Kamera)", "Upload File", "Manual Angka"], 
            horizontal=True
        )
        
        # --- MODE KAMERA ---
        if input_type == "Ambil Foto (Kamera)":
            col_cam_in, col_cam_res = st.columns(2)
            
            with col_cam_in:
                new_code_cam = st.text_input(
                    "Masukkan Kode Warna Baru (Wajib)", 
                    key="code_cam",
                    placeholder="Contoh: RED-001"
                )
                st.write("Arahkan benang ke tengah kamera & klik 'Take Photo'")
                
                use_preprocessing_cam = st.checkbox(
                    "Gunakan Preprocessing Enhanced", 
                    value=True
                )
                
                camera_file = st.camera_input("Kamera Input", label_visibility="collapsed")

            with col_cam_res:
                if camera_file is not None and new_code_cam:
                    bytes_data = camera_file.getvalue()
                    img_cam = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                    
                    if use_preprocessing_cam:
                        img_cam_processed = preprocess_image(img_cam, use_wb=True)
                        st.caption("Preprocessing diterapkan")
                    else:
                        img_cam_processed = img_cam
                    
                    if st.session_state.use_multiple_roi:
                        lab_vals, rgb_avg, coords = extract_color_multiple_roi(img_cam_processed)
                    else:
                        lab_vals, rgb_avg, coords = extract_color_roi_filtered(img_cam_processed)
                    
                    if lab_vals:
                        preview = img_cam.copy()
                        cv2.rectangle(preview, (coords[0], coords[1]), (coords[2], coords[3]), (0,255,0), 3)
                        
                        st.image(preview, channels="BGR", caption="Hasil Capture", use_column_width=True)
                        
                        st.success(f"Terdeteksi LAB: L={lab_vals[0]}, a={lab_vals[1]}, b={lab_vals[2]}")
                        
                        R, G, B = int(rgb_avg[2]), int(rgb_avg[1]), int(rgb_avg[0])
                        color_html = f"""
                        <div style="width:100%; height:60px; background-color:rgb({R}, {G}, {B}); 
                        border:2px solid #000; border-radius:8px; margin:10px 0;"></div>
                        """
                        st.markdown(color_html, unsafe_allow_html=True)
                        
                        if st.button("Simpan ke Database", type="primary", use_container_width=True):
                            succ, msg = save_new_master_data(
                                new_code_cam, 
                                lab_vals[0], lab_vals[1], lab_vals[2], 
                                R, G, B
                            )
                            if succ:
                                st.success(f"Kode {new_code_cam} berhasil disimpan!")
                                time.sleep(1.5)
                                st.cache_data.clear()
                                st.rerun()
                            else:
                                st.error(f"{msg}")
                    else:
                        st.error("Gagal mengekstrak warna")
                        
                elif camera_file is not None and not new_code_cam:
                    st.warning("Mohon isi Kode Warna terlebih dahulu.")

        # --- MODE UPLOAD FILE ---
        elif input_type == "Upload File":
            col_in, col_prev = st.columns(2)
            
            with col_in:
                new_code = st.text_input("Kode Warna", placeholder="Contoh: BLUE-002")
                ref_img = st.file_uploader("Gambar Referensi", type=["jpg", "png"])
                
                use_preprocessing_upload = st.checkbox(
                    "Gunakan Preprocessing Enhanced", 
                    value=True
                )
            
            with col_prev:
                if ref_img and new_code:
                    f_bytes = np.asarray(bytearray(ref_img.read()), dtype=np.uint8)
                    img_ref = cv2.imdecode(f_bytes, 1)
                    
                    if use_preprocessing_upload:
                        img_ref_processed = preprocess_image(img_ref, use_wb=True)
                    else:
                        img_ref_processed = img_ref
                    
                    if st.session_state.use_multiple_roi:
                        lab_vals, rgb_avg, coords = extract_color_multiple_roi(img_ref_processed)
                    else:
                        lab_vals, rgb_avg, coords = extract_color_roi_filtered(img_ref_processed)
                    
                    if lab_vals:
                        preview = img_ref.copy()
                        cv2.rectangle(preview, (coords[0], coords[1]), (coords[2], coords[3]), (0,255,0), 3)
                        st.image(preview, channels="BGR", use_column_width=True)
                        st.write(f"Lab: L={lab_vals[0]}, a={lab_vals[1]}, b={lab_vals[2]}")
                        
                        if st.button("Simpan Data", type="primary"):
                            R, G, B = int(rgb_avg[2]), int(rgb_avg[1]), int(rgb_avg[0])
                            succ, msg = save_new_master_data(
                                new_code, 
                                lab_vals[0], lab_vals[1], lab_vals[2], 
                                R, G, B
                            )
                            if succ:
                                st.success("Tersimpan.")
                                time.sleep(1)
                                st.cache_data.clear()
                                st.rerun()
                            else:
                                st.error(f"{msg}")

        # --- MODE MANUAL ---
        elif input_type == "Manual Angka":
            c1, c2, c3, c4 = st.columns(4)
            with c1: code_m = st.text_input("Kode")
            with c2: L_m = st.number_input("L", 0, 255, 50)
            with c3: a_m = st.number_input("a", 0, 255, 128)
            with c4: b_m = st.number_input("b", 0, 255, 128)
            
            if st.button("Simpan Manual"):
                if code_m:
                    pr_R, pr_G, pr_B = lab_to_bgr_preview(L_m, a_m, b_m)
                    succ, msg = save_new_master_data(code_m, L_m, a_m, b_m, pr_R, pr_G, pr_B)
                    if succ:
                        st.success("Tersimpan.")
                        time.sleep(1)
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error(f"{msg}")
                else:
                    st.error("Kode harus diisi.")
    
    # TAB 3: Analisis
    with tab3:
        st.subheader("Analisis Database")
        
        if not yarn_db.empty:
            col_a1, col_a2 = st.columns(2)
            
            with col_a1:
                st.write("Distribusi Lightness (L)")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(yarn_db['L'], bins=20, color='gray', edgecolor='black')
                ax.set_xlabel('Lightness (L)')
                ax.set_ylabel('Frequency')
                ax.set_title('Distribusi Nilai L')
                st.pyplot(fig)
            
            with col_a2:
                st.write("Plot a vs b")
                fig2, ax2 = plt.subplots(figsize=(8, 4))
                ax2.scatter(yarn_db['a'], yarn_db['b'], c=yarn_db.index, cmap='viridis', s=100, alpha=0.6, edgecolors='black')
                ax2.set_xlabel('a (green-red)')
                ax2.set_ylabel('b (blue-yellow)')
                ax2.set_title('Color Space Distribution')
                ax2.axhline(128, color='gray', linestyle='--', linewidth=0.5)
                ax2.axvline(128, color='gray', linestyle='--', linewidth=0.5)
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2)
            
            st.markdown("---")
            st.write("Statistik Database")
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            
            with col_stat1:
                st.metric("Mean L", f"{yarn_db['L'].mean():.1f}")
            with col_stat2:
                st.metric("Mean a", f"{yarn_db['a'].mean():.1f}")
            with col_stat3:
                st.metric("Mean b", f"{yarn_db['b'].mean():.1f}")
            with col_stat4:
                avg_chroma = np.sqrt(yarn_db['a']**2 + yarn_db['b']**2).mean()
                st.metric("Avg Chroma", f"{avg_chroma:.1f}")
        else:
            st.info("Tidak ada data untuk dianalisis")