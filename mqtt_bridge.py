import time
import json
import os
import paho.mqtt.client as mqtt

# --- KONFIGURASI ---
FILE_TO_WATCH = "riwayat_deteksi.csv" 
MQTT_BROKER = "test.mosquitto.org" 
MQTT_PORT = 1883
MQTT_TOPIC = "yarn-mqtt"

def send_to_mqtt(client, data_row):
    """Mengonversi baris CSV menjadi JSON dan mengirimnya"""
    try:
        parts = data_row.strip().split(',')
        
        # Sesuai urutan CSV: Waktu, Kode_Warna, Jarak_DeltaE, L, a, b, Mode_Input, File_Gambar
        if len(parts) >= 7:
            payload = {
                "waktu": parts[0],
                "kode_warna": parts[1],
                "delta_e": parts[2],
                "lab": {
                    "L": parts[3],
                    "a": parts[4],
                    "b": parts[5]
                },
                "mode": parts[6],
                "file": parts[7] if len(parts) > 7 else "-"
            }
            
            # retain=True membuat broker menyimpan pesan terakhir untuk pelanggan baru
            client.publish(MQTT_TOPIC, json.dumps(payload), retain=True)
            print(f"[SENT] Data {parts[1]} sukses terkirim ke {MQTT_TOPIC}")
    except Exception as e:
        print(f"Error memproses baris: {e}")

def monitor_csv():
    # Inisialisasi Client
    client = mqtt.Client()
    
    try:
        print(f"Menghubungkan ke Broker: {MQTT_BROKER}...")
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start() # Jalankan loop di background
        print(f"--- MQTT BRIDGE AKTIF (Public Broker) ---")
        print(f"Memantau: {FILE_TO_WATCH} | Topik: {MQTT_TOPIC}")
        print("-" * 45)
    except Exception as e:
        print(f"Gagal terhubung ke Broker: {e}")
        return

    # Tentukan posisi awal (agar tidak mengirim data lama)
    last_pos = os.path.getsize(FILE_TO_WATCH) if os.path.exists(FILE_TO_WATCH) else 0

    try:
        while True:
            if os.path.exists(FILE_TO_WATCH):
                current_size = os.path.getsize(FILE_TO_WATCH)
                
                if current_size > last_pos:
                    with open(FILE_TO_WATCH, 'r') as f:
                        f.seek(last_pos)
                        new_lines = f.readlines()
                        for line in new_lines:
                            if line.strip() and not line.startswith("Waktu"):
                                send_to_mqtt(client, line)
                    last_pos = current_size
            
            time.sleep(1) # Cek file setiap detik
    except KeyboardInterrupt:
        print("\nMematikan Bridge...")
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    monitor_csv()