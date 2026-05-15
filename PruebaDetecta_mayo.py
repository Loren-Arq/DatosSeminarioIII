import serial
import wave
import os
import numpy as np
import time
from datetime import datetime

SERIAL_PORT     = 'COM3'
BAUD_RATE       = 921600
SAMPLE_RATE     = 16000
RECORD_SECONDS  = 4
OUTPUT_DIR      = 'AudiosMayo/Audio'
FACTOR_AMP      = 10.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

def leer_lineas_meta(ser):
    """Lee líneas de texto hasta encontrar START_AUDIO."""
    meta = {}
    while True:
        raw = ser.readline()
        line = raw.decode('utf-8', errors='ignore').strip()

        if line.startswith("OBJ:"):
            meta['distancia'] = line.split(":")[1]
        elif line.startswith("TIME_MS:"):
            meta['tiempo_arduino_ms'] = int(line.split(":")[1])
        elif line.startswith("LAT_SENSOR_MS:"):
            meta['lat_sensor_ms'] = int(line.split(":")[1])
        elif line.startswith("LAT_TRIGGER_MS:"):
            meta['lat_trigger_ms'] = int(line.split(":")[1])
        elif line.startswith("LAT_TOTAL_MS:"):
            meta['lat_total_esp_ms'] = int(line.split(":")[1])
        elif "START_AUDIO" in line:
            return meta   # ← sale del loop solo aquí

def leer_audio_binario(ser, bytes_esperados):
    """
    Lee exactamente bytes_esperados bytes del stream binario,
    sin confundirlos con líneas de texto.
    """
    buffer = bytearray()
    t_inicio = time.perf_counter()

    while len(buffer) < bytes_esperados:
        pendientes = bytes_esperados - len(buffer)
        # Lee en trozos para no bloquear indefinidamente
        chunk = ser.read(min(pendientes, 4096))
        if chunk:
            buffer.extend(chunk)

    return bytes(buffer), time.perf_counter() - t_inicio

def leer_footer(ser):
    """Consume el marcador nulo + END_AUDIO + métricas finales."""
    footer = {}
    # Consumir los 4 bytes nulos de separación
    ser.read(4)
    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if not line:
            continue
        if "END_AUDIO" in line:
            break
        if line.startswith("LAT_CAPTURE_TOTAL_MS:"):
            footer['lat_capture_total_esp'] = int(line.split(":")[1])
    return footer

def record_from_serial():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=5)
        print(f"Conectado a {SERIAL_PORT}. Esperando detección...")
    except Exception as e:
        print(f"Error: {e}")
        return

    bytes_esperados = SAMPLE_RATE * RECORD_SECONDS * 2   # 128 000 bytes

    while True:
        try:
            # ── 1. Esperar y leer toda la metadata de texto ──────────────────
            meta = leer_lineas_meta(ser)
            t_start_pc = time.perf_counter()

            print(f"\n🎯 Detección a {meta.get('distancia','?')} mm")
            print(f"   [ESP32] Sensor tardó    : {meta.get('lat_sensor_ms', 0)} ms")
            print(f"   [ESP32] Overhead trigger: {meta.get('lat_trigger_ms', 0)} ms")
            print(f"   [ESP32] Latencia total  : {meta.get('lat_total_esp_ms', 0)} ms")
            print("📥 Recibiendo audio puro...")

            # ── 2. Leer exactamente N bytes binarios ─────────────────────────
            raw_audio, lat_rx = leer_audio_binario(ser, bytes_esperados)

            # ── 3. Leer footer (marcador nulo + END_AUDIO + métricas) ────────
            footer = leer_footer(ser)

            # ── 4. Procesar ──────────────────────────────────────────────────
            t_proc_start = time.perf_counter()

            samples = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32)
            samples *= FACTOR_AMP
            samples  = np.clip(samples, -32768, 32767).astype(np.int16)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename  = os.path.join(OUTPUT_DIR, f'audio_{timestamp}.wav')

            with wave.open(filename, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(SAMPLE_RATE)
                wav.writeframes(samples.tobytes())

            lat_proc   = (time.perf_counter() - t_proc_start) * 1000
            lat_e2e    = (time.perf_counter() - t_start_pc)   * 1000
            lat_real   = meta.get('lat_total_esp_ms', 0) + lat_e2e

            sep = "─" * 46
            print(f"\n{sep}")
            print(f"✅  {os.path.basename(filename)}")
            print(f"{sep}")
            print(f"  Muestras recibidas        : {len(samples):>8}")
            print(f"  Duración real             : {len(samples)/SAMPLE_RATE:>7.2f} s")
            print(f"  [ESP32] Captura total     : {footer.get('lat_capture_total_esp',0):>8} ms")
            print(f"  Transmisión serial        : {lat_rx*1000:>8.1f} ms")
            print(f"  Procesamiento + guardado  : {lat_proc:>8.1f} ms")
            print(f"  ⏱  Latencia E2E real      : {lat_real:>8.1f} ms")
            print(f"{sep}\n")

        except KeyboardInterrupt:
            print("\nPrograma detenido.")
            break

    ser.close()

if __name__ == "__main__":
    record_from_serial()