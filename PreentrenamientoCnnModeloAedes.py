import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from scipy.signal import butter, lfilter # IMPORTANTE AÑADIR ESTO

# --- NUEVAS FUNCIONES DE LIMPIEZA ---
def filtro_pasa_alta(data, sr, cutoff=300, order=6):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1: return data
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return lfilter(b, a, data)
# 1. Configuración de rutas y parámetros
DATA_PATH = 'AudiosMayo' # Carpeta que contiene 'SiAedes' y 'NoAedes'
IMG_SIZE = (128, 128) # Tamaño al que redimensionaremos los espectrogramas
EPOCHS = 50

def load_data(data_path):
    images = []
    labels = []
    classes = ['NoAedes_Generados', 'SiAedes_Generados']
    img_size = (128, 128)
    
    for idx, label_name in enumerate(classes):
        folder = os.path.join(data_path, label_name)
        for file in os.listdir(folder):
            if file.endswith('.wav'):
                path = os.path.join(folder, file)
                
                # 1. Cargar audio
                y, sr = librosa.load(path, duration=3.0, sr=None)
                
                # 2. APLICAR FILTRO Y NORMALIZACIÓN (Igual que en la detección)
                y = filtro_pasa_alta(y, sr)
                if np.max(np.abs(y)) > 0:
                    y = librosa.util.normalize(y)
                
                # 3. Convertir a Espectrograma de Mel con rangos específicos
                # Usamos fmin y fmax para que la CNN se enfoque en el mosquito
                mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmin=200, fmax=2000)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                
                # 4. Normalización de la imagen (Min-Max)
                img = tf.image.resize(mel_spec_db[..., np.newaxis], img_size).numpy()
                if np.max(img) - np.min(img) != 0:
                    img = (img - np.min(img)) / (np.max(img) - np.min(img))
                
                images.append(img)
                labels.append(idx)
                
    return np.array(images), np.array(labels)

# El resto del código de arquitectura y model.fit se mantiene igual...

# 2. Preparar los datos
print("Cargando y procesando audios...")
X, y = load_data(DATA_PATH)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. Definir la arquitectura de la CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid') # Salida binaria (Probabilidad)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Entrenar y Guardar
print("Iniciando entrenamiento...")
model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_test, y_test))

model.save('mi_modelo_aedes.h5')
print("Modelo guardado exitosamente como 'mi_modelo_aedes.h5'")
