import streamlit as st
import cv2
import numpy as np
import pandas as pd
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
import av
import os
import time
from datetime import datetime
import queue

# --- 1. KONFIGURASI AWAL (MINIMALIS) ---
st.set_page_config(page_title="Yarn Master System", layout="wide")

DATABASE_FILE = "database_yarn.csv"
HISTORY_FILE = "riwayat_deteksi.csv"
SNAPSHOT_FOLDER = "snapshots" 
ROI_SIZE = 80

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

# Queue untuk komunikasi data
if "detection_queue" not in st.session_state:
    st.session_state.detection_queue = queue.Queue()

# --- 2. FUNGSI LOGIKA (BACKEND) ---

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
    
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h: return None, None, None

    roi = image_bgr[y1:y2, x1:x2]
    avg_bgr = np.mean(roi, axis=(0, 1))
    
    pixel_bgr = np.uint8([[avg_bgr]])
    pixel_lab = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2Lab)[0][0]
    
    L, a, b = int(pixel_lab[0]), int(pixel_lab[1]), int(pixel_lab[2])
    return (L, a, b), avg_bgr, (x1, y1, x2, y2)

def find_closest(detected_lab, db_df):
    if db_df.empty: return None, 0
    l_diff = db_df['L'] - detected_lab[0]
    a_diff = db_df['a'] - detected_lab[1]
    b_diff = db_df['b'] - detected_lab[2]
    distances = np.sqrt(l_diff**2 + a_diff**2 + b_diff**2)
    min_idx = distances.idxmin()
    return db_df.loc[min_idx], distances[min_idx]

# --- 3. CALLBACK WEBRTC ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.resize(img, (640, 480))
    
    lab_vals, _, coords = extract_color_roi(img)
    
    if lab_vals:
        x1, y1, x2, y2 = coords
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        try:
            db_df = pd.read_csv(DATABASE_FILE)
            best_match, dist = find_closest(lab_vals, db_df)
            
            if best_match is not None:
                kode = str(best_match['Kode_Warna'])
                text = f"{kode} ({dist:.1f})"
                cv2.rectangle(img, (x1, y1 - 25), (x1 + 200, y1), (0, 0, 0), -1)
                cv2.putText(img, text, (x1+5, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                if not st.session_state.detection_queue.full():
                    try:
                        while not st.session_state.detection_queue.empty():
                            st.session_state.detection_queue.get_nowait()
                        st.session_state.detection_queue.put({
                            "kode": kode,
                            "dist": dist,
                            "lab": lab_vals,
                            "image": img 
                        })
                    except: pass
        except: pass
            
    return av.VideoFrame.from_ndarray(img, format="bgr24")


# --- 4. TAMPILAN ANTARMUKA ---

yarn_db = load_database(DATABASE_FILE)

st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Halaman", [
    "Deteksi Realtime", 
    "Deteksi Upload",
    "Database Manager"
])
st.sidebar.markdown("---")

# ====================================================
# HALAMAN 1: DETEKSI REALTIME
# ====================================================
if menu == "Deteksi Realtime":
    st.title("Deteksi Realtime")
    st.write("Arahkan kamera ke objek. Klik simpan untuk merekam data.")
    
    col_cam, col_ctrl = st.columns([2, 1])
    
    with col_cam:
        rtc_cfg = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        webrtc_streamer(
            key="realtime-view",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_cfg,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )

    with col_ctrl:
        st.subheader("Kontrol")
        if st.button("Simpan Data & Gambar", type="primary", use_container_width=True):
            if not st.session_state.detection_queue.empty():
                try:
                    data_packet = st.session_state.detection_queue.get()
                    kode = data_packet['kode']
                    dist = data_packet['dist']
                    lab = data_packet['lab']
                    img_capture = data_packet['image']
                    
                    filename_img = save_snapshot(img_capture, kode)
                    save_to_history(kode, dist, lab, "Realtime", filename_img)
                    
                    st.success(f"Tersimpan: {kode}")
                    st.caption(f"File: {filename_img}")
                except Exception as e:
                    st.error(f"Gagal: {e}")
            else:
                st.warning("Belum ada objek terdeteksi.")

        st.markdown("---")
        st.write("Riwayat Terakhir:")
        if os.path.exists(HISTORY_FILE):
            try:
                df_hist = pd.read_csv(HISTORY_FILE)
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
            st.image(img, channels="BGR", use_column_width=True)
            
        with col_res:
            if lab_res:
                match_row, dist = find_closest(lab_res, yarn_db)
                if match_row is not None:
                    st.subheader(f"Hasil: {match_row['Kode_Warna']}")
                    st.write(f"Delta E: {dist:.2f}")
                    st.code(f"L:{lab_res[0]} a:{lab_res[1]} b:{lab_res[2]}")
                    
                    if st.button("Simpan Hasil"):
                        filename_img = save_snapshot(img, match_row['Kode_Warna'])
                        save_to_history(match_row['Kode_Warna'], dist, lab_res, "Upload", filename_img)
                        st.success("Data tersimpan.")
                else:
                    st.warning("Tidak cocok.")

# ====================================================
# HALAMAN 3: DATABASE MANAGER
# ====================================================
elif menu == "Database Manager":
    st.title("Database Manager")
    
    tab1, tab2 = st.tabs(["Lihat Data", "Tambah Data"])
    
    with tab1:
        if not yarn_db.empty:
            st.dataframe(yarn_db, use_container_width=True, height=500)
        else:
            st.info("Database kosong.")

    with tab2:
        st.write("### Tambah Data Master Baru")
        # Menambahkan Opsi "Ambil Foto (Kamera)" di sini
        input_type = st.radio("Metode Input", ["Ambil Foto (Kamera)", "Upload File", "Manual Angka"], horizontal=True)
        
        # --- MODE 1: KAMERA REALTIME (BARU) ---
        if input_type == "Ambil Foto (Kamera)":
            col_cam_in, col_cam_res = st.columns(2)
            
            with col_cam_in:
                new_code_cam = st.text_input("1. Masukkan Kode Warna Baru (Wajib)", key="code_cam")
                st.write("2. Arahkan benang ke tengah kamera & klik 'Take Photo'")
                # Widget Kamera Native Streamlit
                camera_file = st.camera_input("Kamera Input", label_visibility="collapsed")

            with col_cam_res:
                if camera_file is not None and new_code_cam:
                    # Proses gambar dari kamera
                    bytes_data = camera_file.getvalue()
                    img_cam = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                    
                    # Ekstrak Warna
                    lab_vals, rgb_avg, coords = extract_color_roi(img_cam)
                    
                    if lab_vals:
                        # Gambar kotak ROI di hasil foto
                        preview = img_cam.copy()
                        cv2.rectangle(preview, (coords[0], coords[1]), (coords[2], coords[3]), (0,255,0), 2)
                        
                        st.image(preview, channels="BGR", caption="Hasil Capture", width=250)
                        st.info(f"Terdeteksi LAB: {lab_vals}")
                        
                        if st.button("Simpan ke Database", type="primary"):
                            R, G, B = int(rgb_avg[2]), int(rgb_avg[1]), int(rgb_avg[0])
                            succ, msg = save_new_master_data(new_code_cam, lab_vals[0], lab_vals[1], lab_vals[2], R, G, B)
                            if succ:
                                st.success(f"Kode {new_code_cam} berhasil disimpan!")
                                time.sleep(1.5)
                                st.cache_data.clear()
                                st.rerun()
                            else:
                                st.error(msg)
                elif camera_file is not None and not new_code_cam:
                    st.warning("Mohon isi Kode Warna terlebih dahulu di sebelah kiri.")

        # --- MODE 2: UPLOAD FILE (LAMA) ---
        elif input_type == "Upload File":
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
                                st.success("Tersimpan.")
                                time.sleep(1)
                                st.cache_data.clear()
                                st.rerun()
                            else:
                                st.error(msg)

        # --- MODE 3: MANUAL ANGKA (LAMA) ---
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
                        st.error(msg)
                else:
                    st.error("Kode harus diisi.")