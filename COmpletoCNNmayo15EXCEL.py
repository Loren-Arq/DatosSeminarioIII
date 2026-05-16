import os
import librosa
import numpy as np
import tensorflow as tf
import pandas as pd
from scipy.signal import butter, lfilter

# --- 1. CONFIGURACIÓN DE RUTAS ---
MODEL_PATH = 'mi_modelo_aedes.h5'
INPUT_FOLDER = 'AudiosMayo/Audio' 
EXCEL_NAME = 'Resultados_Aedes.xlsx'

# --- 2. DEFINICIÓN DE FUNCIONES DE LIMPIEZA ---

def filtro_pasa_alta(data, sr):
    cutoff = 300
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1:
        return data
    b, a = butter(6, normal_cutoff, btype='high', analog=False)
    return lfilter(b, a, data)

def procesar_audio_aedes(y, sr):
    # Filtrar ruidos graves y normalizar volumen
    y_filtrado = filtro_pasa_alta(y, sr)
    if np.max(np.abs(y_filtrado)) > 0:
        y_norm = librosa.util.normalize(y_filtrado)
    else:
        y_norm = y_filtrado
    return y_norm

def analizar_mosquito(file_path, model):
    # Cargar audio original
    y_raw, sr = librosa.load(file_path, sr=None)
    
    # Preprocesar (Limpieza y Amplificación)
    y = procesar_audio_aedes(y_raw, sr)
    
    # --- CÁLCULO DE FRECUENCIA DOMINANTE ---
    S = np.abs(librosa.stft(y))
    f = librosa.fft_frequencies(sr=sr)
    
    # Promediamos el espectro en el tiempo y filtramos el rango (200-2000Hz)
    S_mean = np.mean(S, axis=1)
    mask = (f >= 200) & (f <= 2000)
    
    if np.any(mask):
        f_sub = f[mask]
        S_sub = S_mean[mask]
        freq_dominante = f_sub[np.argmax(S_sub)]
    else:
        freq_dominante = 0.0

    # --- PREDICCIÓN DE PROBABILIDAD (CNN) ---
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, fmin=200, fmax=2000, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Redimensionar para la CNN (128x128)
    img = tf.image.resize(mel_spec_db[..., np.newaxis], (128, 128)).numpy()
    if np.max(img) - np.min(img) != 0:
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
    
    img = np.expand_dims(img, axis=0)
    # Obtenemos el valor escalar de la probabilidad inicial del modelo
    probabilidad = model.predict(img, verbose=0)[0][0]
    
    # --- LATENCIA ---
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    latencia = librosa.times_like(onset_env, sr=sr)[np.argmax(onset_env)]
    
    # === FILTRO DE VALIDACIÓN GENERAL ===
    # El rango biológico estándar para una hembra de Aedes aegypti es de 400-600Hz.
    # Ajustamos umbrales tolerantes entre 380 y 620 Hz para evitar falsos positivos de cualquier audio.
    FREC_MIN_AEDES = 380.0
    FREC_MAX_AEDES = 620.0
    
    if (freq_dominante < FREC_MIN_AEDES) or (freq_dominante > FREC_MAX_AEDES):
        probabilidad = 0.0

    return probabilidad, freq_dominante, latencia

# --- 3. PROCESO PRINCIPAL ---

# Cargar el modelo
if not os.path.exists(MODEL_PATH):
    print(f"Error: No se encuentra el modelo en {MODEL_PATH}")
    exit()

model = tf.keras.models.load_model(MODEL_PATH)

resultados = []
contador_evento = 1

print(f"{'No':<4} | {'Archivo':<25} | {'Probabilidad':<12} | {'Frecuencia':<12}")
print("-" * 65)

if not os.path.exists(INPUT_FOLDER):
    print(f"Error: La carpeta '{INPUT_FOLDER}' no existe.")
else:
    for archivo in os.listdir(INPUT_FOLDER):
        if archivo.lower().endswith('.wav'):
            ruta_completa = os.path.join(INPUT_FOLDER, archivo)
            
            try:
                prob, freq, lat = analizar_mosquito(ruta_completa, model)
                
                # Guardar en la lista para el Excel
                resultados.append({
                    'No de evento': contador_evento,
                    'Nombre archivo': archivo,
                    'Probabilidad': f"{prob:.2%}",
                    'Latencia': f"{lat:.3f} s",
                    'Frecuencia Central': f"{freq:.2f} Hz"
                })
                
                print(f"{contador_evento:<4} | {archivo[:25]:<25} | {prob:.2%}      | {freq:>8.2f} Hz")
                contador_evento += 1
                
            except Exception as e:
                print(f"Error procesando {archivo}: {e}")

    # --- 4. EXPORTAR A EXCEL ---
    if resultados:
        df = pd.DataFrame(resultados)
        df.to_excel(EXCEL_NAME, index=False)
        print(f"\n✅ Proceso terminado. Datos guardados en: {EXCEL_NAME}")
    else:
        print("\n❌ No se encontraron archivos .wav para procesar.")
