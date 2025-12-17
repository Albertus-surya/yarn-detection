import streamlit as st
import cv2
import numpy as np
import pandas as pd
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode, VideoProcessorBase
import av
import os
import time
from datetime import datetime
import threading

# --- 1. KONFIGURASI AWAL ---
st.set_page_config(page_title="Yarn Master System", layout="wide")

DATABASE_FILE = "database_yarn.csv"
HISTORY_FILE = "riwayat_deteksi.csv"
SNAPSHOT_FOLDER = "snapshots"

# KONFIGURASI DETEKSI
ROI_SIZE = 50 
THRESHOLD = 25.0  # batasan

# Buat folder snapshot jika belum ada
if not os.path.exists(SNAPSHOT_FOLDER):
    os.makedirs(SNAPSHOT_FOLDER)

# Inisialisasi File CSV
if not os.path.exists(DATABASE_FILE):
    with open(DATABASE_FILE, 'w') as f:
        f.write("Kode_Warna,L,a,b,R_preview,G_preview,B_preview\n")

if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'w') as f:
        f.write("Waktu,Kode_Warna,Jarak_DeltaE,L,a,b,Mode_Input,File_Gambar\n")

# --- 2. FUNGSI LOGIKA ---

@st.cache_data
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
    """Menyimpan file gambar JPG"""
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{kode}_{timestamp_str}.jpg"
    filepath = os.path.join(SNAPSHOT_FOLDER, filename)
    cv2.imwrite(filepath, image_bgr)
    return filename

def save_to_history(kode, distance, lab_values, source_mode, image_filename="-"):
    """Mencatat log ke CSV"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(HISTORY_FILE, mode='a', newline='\n') as f:
        row = f"{timestamp},{kode},{distance:.2f},{lab_values[0]},{lab_values[1]},{lab_values[2]},{source_mode},{image_filename}\n"
        f.write(row)

def save_new_master_data(code, l, a, b, r, g, b_val):
    try:
        curr_db = pd.read_csv(DATABASE_FILE)
        if code in curr_db['Kode_Warna'].values.astype(str):
            return False, f"Kode '{code}' sudah ada."
        
        with open(DATABASE_FILE, mode='a', newline='\n') as f:
            f.write(f"{code},{l},{a},{b},{r},{g},{b_val}\n")
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

def extract_color_roi(image_bgr):
    h, w, _ = image_bgr.shape
    cx, cy = w // 2, h // 2
    x1, y1, x2, y2 = cx - ROI_SIZE, cy - ROI_SIZE, cx + ROI_SIZE, cy + ROI_SIZE
    
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h: 
        return None, None, None

    roi = image_bgr[y1:y2, x1:x2]
    avg_bgr = np.mean(roi, axis=(0, 1))
    
    pixel_bgr = np.uint8([[avg_bgr]])
    pixel_lab = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2Lab)[0][0]
    
    L, a, b = int(pixel_lab[0]), int(pixel_lab[1]), int(pixel_lab[2])
    return (L, a, b), avg_bgr, (x1, y1, x2, y2)

# --- FUNGSI PENCARIAN  TOP 3 ---
def find_top_matches(detected_lab, db_df, top_n=3):
    """Mencari Top-N match terdekat berdasarkan Delta E"""
    if db_df.empty: 
        return []
    
    # Hitung Delta E ke semua data
    l_diff = db_df['L'] - detected_lab[0]
    a_diff = db_df['a'] - detected_lab[1]
    b_diff = db_df['b'] - detected_lab[2]
    distances = np.sqrt(l_diff**2 + a_diff**2 + b_diff**2)
    
    # Ambil index dari jarak terkecil ke terbesar
    sorted_indices = distances.nsmallest(top_n).index
    
    results = []
    for idx in sorted_indices:
        dist = distances[idx]
        # Hanya masukkan jika masuk dalam THRESHOLD
        if dist <= THRESHOLD:
            row_data = db_df.loc[idx]
            results.append({
                "kode": str(row_data['Kode_Warna']),
                "dist": dist,
                "data": row_data
            })
            
    return results

# Cache database di memori
_cached_db = None
_cached_db_time = 0

def get_cached_database():
    global _cached_db, _cached_db_time
    current_time = time.time()
    if _cached_db is None or (current_time - _cached_db_time) > 1:
        try:
            _cached_db = pd.read_csv(DATABASE_FILE)
        except:
            _cached_db = pd.DataFrame()
    return _cached_db

# --- 3. VIDEO PROCESSOR CLASS UPDATE TOP 3 ---
class YarnDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_detection = None
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (640, 480))
        
        lab_vals, _, coords = extract_color_roi(img)
        
        if lab_vals:
            x1, y1, x2, y2 = coords
            # Gambar Kotak ROI Hijau
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            try:
                db_df = get_cached_database()
                # Panggil fungsi Top 3
                matches = find_top_matches(lab_vals, db_df, top_n=3)
                
                if matches:
                    # Ambil Match Terbaik (Index 0)
                    best = matches[0]
                    kode_utama = best['kode']
                    dist_utama = best['dist']
                    
                    # TAMPILAN UTAMA (Background Hitam + Teks Hijau)
                    text_main = f"{kode_utama} ({dist_utama:.1f})"
                    cv2.rectangle(img, (x1, y1 - 35), (x1 + 200, y1), (0, 0, 0), -1)
                    cv2.putText(img, text_main, (x1+5, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # TAMPILAN ALTERNATIF (Jika ada match ke-2)
                    if len(matches) > 1:
                        # Tampilkan di bawah kotak
                        y_offset = y2 + 20
                        for i in range(1, len(matches)):
                            alt = matches[i]
                            # Hanya tampilkan jika jaraknya beda tipis (misal selisih < 10) atau memang dekat
                            text_alt = f"Alt: {alt['kode']} ({alt['dist']:.1f})"
                            cv2.putText(img, text_alt, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                            y_offset += 20
                    
                    # Update detection data
                    with self.lock:
                        self.latest_detection = {
                            "matches": matches, # Simpan semua list match
                            "lab": lab_vals,
                            "image": img.copy(),
                            "timestamp": time.time()
                        }
            except Exception as e:
                pass
                
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def get_latest_detection(self):
        with self.lock:
            return self.latest_detection

# --- 4. TAMPILAN ANTARMUKA ---

yarn_db = load_database(DATABASE_FILE)

st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Halaman", [
    "Deteksi Realtime", 
    "Deteksi Upload",
    "Database Manager"
])
st.sidebar.markdown("---")
st.sidebar.info(f"**Info Sistem**\n\nThreshold: {THRESHOLD}\nROI Size: {ROI_SIZE}")

# ====================================================
# HALAMAN 1: DETEKSI REALTIME (CLASS-BASED)
# ====================================================
if menu == "Deteksi Realtime":
    st.title("Deteksi Realtime")
    st.write("Arahkan kamera ke objek. Sistem akan menampilkan kode utama dan alternatif jika ada.")
    
    # Initialize processor in session state
    if "video_processor" not in st.session_state:
        st.session_state.video_processor = None
    
    col_cam, col_ctrl = st.columns([2, 1])
    
    with col_cam:
        rtc_cfg = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        webrtc_ctx = webrtc_streamer(
            key="realtime-yarn-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_cfg,
            video_processor_factory=YarnDetectionProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )
        
        if webrtc_ctx.video_processor:
            st.session_state.video_processor = webrtc_ctx.video_processor

    with col_ctrl:
        st.subheader("Hasil Deteksi")
        
        info_placeholder = st.empty()
        
        # Get detection data
        current_data = None
        if st.session_state.video_processor:
            current_data = st.session_state.video_processor.get_latest_detection()
        
        # Display Info
        if current_data and (time.time() - current_data['timestamp'] < 2):
            matches = current_data['matches']
            lab = current_data['lab']
            
            # Tampilkan Match Utama
            best = matches[0]
            st.success(f"### 🎯 Utama: {best['kode']}")
            st.write(f"Delta E: **{best['dist']:.2f}**")
            st.caption(f"Lab Input: {lab}")
            
            # Tampilkan Alternatif (Jika ada)
            if len(matches) > 1:
                st.markdown("---")
                st.write("**Kandidat Lain:**")
                for i in range(1, len(matches)):
                    alt = matches[i]
                    st.warning(f"🔹 **{alt['kode']}** (Diff: {alt['dist']:.2f})")
            
        else:
            info_placeholder.info("Menunggu objek...")
        
        st.markdown("---")
        
        # Tombol Simpan
        if st.button("Simpan Data & Gambar", type="primary", use_container_width=True):
            if current_data and (time.time() - current_data['timestamp'] < 2):
                try:
                    # Ambil data terbaik (Index 0)
                    best = current_data['matches'][0]
                    kode = best['kode']
                    dist = best['dist']
                    lab = current_data['lab']
                    img_capture = current_data['image']
                    
                    with st.spinner("Menyimpan data..."):
                        filename_img = save_snapshot(img_capture, kode)
                        save_to_history(kode, dist, lab, "Realtime", filename_img)
                        time.sleep(0.3)
                    
                    st.success(f"✅ **Tersimpan: {kode}**")
                    time.sleep(1.5)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Gagal menyimpan: {e}")
            else:
                st.error("⚠️ Tidak ada objek terdeteksi untuk disimpan.")

        # Riwayat Mini
        with st.expander("Riwayat Terakhir"):
            if os.path.exists(HISTORY_FILE):
                try:
                    df_hist = pd.read_csv(HISTORY_FILE)
                    if not df_hist.empty:
                        st.dataframe(df_hist[['Waktu', 'Kode_Warna', 'Jarak_DeltaE']].tail(5), hide_index=True)
                except: pass

# ====================================================
# HALAMAN 2: DETEKSI UPLOAD
# ====================================================
elif menu == "Deteksi Upload":
    st.title("Deteksi Upload")
    
    uploaded_file = st.file_uploader("Upload gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        col_img, col_res = st.columns(2)
        
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        lab_res, _, coords = extract_color_roi(img)
        
        with col_img:
            if coords:
                cv2.rectangle(img, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)
            st.image(img, channels="BGR", use_container_width=True)
            
        with col_res:
            if lab_res:
                st.info(f"🔍 Terbaca Lab: {lab_res}")
                
                # Gunakan fungsi Top Matches
                matches = find_top_matches(lab_res, yarn_db, top_n=3)
                
                if matches:
                    # Tampilkan Terbaik
                    best = matches[0]
                    st.subheader(f"✅ Hasil Terbaik: {best['kode']}")
                    st.metric("Jarak (Delta E)", f"{best['dist']:.2f}")
                    
                    # Tampilkan Alternatif
                    if len(matches) > 1:
                        st.markdown("---")
                        st.write("📋 **Kemungkinan Lain:**")
                        for i in range(1, len(matches)):
                            alt = matches[i]
                            st.write(f"- **{alt['kode']}** (Jarak: {alt['dist']:.2f})")
                    
                    st.markdown("---")
                    if st.button("Simpan Hasil Terbaik"):
                        filename_img = save_snapshot(img, best['kode'])
                        save_to_history(best['kode'], best['dist'], lab_res, "Upload", filename_img)
                        st.success("✅ Data tersimpan.")
                        time.sleep(1)
                        st.rerun()
                else:
                    st.error(f"⚠️ Tidak ada kecocokan (Semua di atas Threshold {THRESHOLD}).")
                    st.write("Coba pastikan pencahayaan cukup terang.")

# ====================================================
# HALAMAN 3: DATABASE MANAGER
# ====================================================
elif menu == "Database Manager":
    st.title("Database Manager")
    
    tab1, tab2 = st.tabs(["📋 Lihat Data", "➕ Tambah Data"])
    
    with tab1:
        if not yarn_db.empty:
            st.dataframe(yarn_db, use_container_width=True, height=500)
        else:
            st.info("Database kosong.")

    with tab2:
        st.write("Tambah Referensi Warna")
        input_type = st.radio("Metode Input", ["Upload Gambar", "Manual Angka"])
        
        if input_type == "Upload Gambar":
            col_in, col_prev = st.columns(2)
            with col_in:
                new_code = st.text_input("Kode Warna")
                ref_img = st.file_uploader("Gambar Referensi", type=["jpg", "png"])
            
            with col_prev:
                if ref_img and new_code:
                    f_bytes = np.asarray(bytearray(ref_img.read()), dtype=np.uint8)
                    img_ref = cv2.imdecode(f_bytes, 1)
                    lab_vals, rgb_avg, coords = extract_color_roi(img_ref)
                    
                    if lab_vals:
                        preview = img_ref.copy()
                        cv2.rectangle(preview, (coords[0], coords[1]), (coords[2], coords[3]), (0,255,0), 2)
                        st.image(preview, channels="BGR", width=200)
                        st.write(f"Lab: {lab_vals}")
                        
                        if st.button("Simpan Data"):
                            R, G, B = int(rgb_avg[2]), int(rgb_avg[1]), int(rgb_avg[0])
                            succ, msg = save_new_master_data(new_code, lab_vals[0], lab_vals[1], lab_vals[2], R, G, B)
                            if succ:
                                st.success("✅ Tersimpan.")
                                time.sleep(1)
                                st.cache_data.clear()
                                st.rerun()
                            else:
                                st.error(msg)
        
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
                        st.success("✅ Tersimpan.")
                        time.sleep(1)
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.error("❌ Kode harus diisi.")