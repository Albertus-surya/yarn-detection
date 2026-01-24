import time
import json
import os
import paho.mqtt.client as mqtt

# --- KONFIGURASI ---
FILE_TO_WATCH = "scan_history.csv" 
MQTT_BROKER = "test.mosquitto.org" 
MQTT_PORT = 1883
MQTT_TOPIC = "yarn-mqtt"

def send_to_mqtt(client, data_row):
    """Mengonversi baris CSV menjadi JSON dan mengirimnya ke MQTT"""
    try:
        parts = data_row.strip().split(',')

        # Pastikan jumlah kolom sesuai
        if len(parts) >= 9:
            payload = {
                "timestamp": parts[0],
                "label": parts[1],
                "confidence": float(parts[2]),
                "mean_lab": {
                    "L": float(parts[3]),
                    "a": float(parts[4]),
                    "b": float(parts[5])
                },
                "std_lab": {
                    "L": float(parts[6]),
                    "a": float(parts[7]),
                    "b": float(parts[8])
                }
            }

            client.publish(
                MQTT_TOPIC,
                json.dumps(payload),
                retain=True
            )

            print(f"[SENT] Label {parts[1]} | Confidence {parts[2]} terkirim")

    except Exception as e:
        print(f"[ERROR] Gagal memproses baris: {e}")

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