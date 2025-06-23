# entrenador.py
import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

# Ruta a la carpeta de audios
AUDIO_DIR = "audios"

# Emociones válidas
CLASES_VALIDAS = ["ira", "calma", "tristeza", "panico"]

def calcular_fft(segmento, sr):
    N = len(segmento)
    fft = np.fft.fft(segmento)
    magnitud = np.abs(fft)[:N // 2]
    freqs = np.fft.fftfreq(N, 1/sr)[:N // 2]
    return freqs, magnitud

def extraer_caracteristicas(y, sr):
    if len(y) < sr*2:
        y = np.pad(y, (0, sr*2 - len(y)))
    else:
        y = y[:sr*2]

    y = y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y
    freqs, magnitud = calcular_fft(y, sr)

    energia = np.sum(y ** 2)
    cero_cruces = librosa.feature.zero_crossing_rate(y)[0].mean()
    centroide = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0].mean()

    bandas = {
        'baja': np.sum(magnitud[(freqs >= 0) & (freqs < 300)]),
        'media': np.sum(magnitud[(freqs >= 300) & (freqs < 1500)]),
        'alta': np.sum(magnitud[(freqs >= 1500)])
    }

    proporcion_altas = bandas['alta'] / (bandas['media'] + 1e-6)
    proporcion_media = bandas['media'] / (bandas['baja'] + 1e-6)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = mfccs.mean(axis=1)

    return list(mfccs_mean) + [
        energia,
        cero_cruces,
        centroide,
        rolloff,
        proporcion_altas,
        proporcion_media
    ]

def cargar_datos():
    X, y = [], []
    for archivo in os.listdir(AUDIO_DIR):
        if not archivo.endswith(".wav"):
            continue

        ruta = os.path.join(AUDIO_DIR, archivo)
        nombre = archivo.lower()

        emocion = None
        for clase in CLASES_VALIDAS:
            if clase in nombre:
                emocion = clase
                break
        if emocion is None:
            print(f"⚠️  Saltando {archivo}: no se detectó clase.")
            continue

        y_audio, sr = librosa.load(ruta, sr=None)
        caracteristicas = extraer_caracteristicas(y_audio, sr)
        X.append(caracteristicas)
        y.append(emocion)

    return np.array(X), np.array(y)

def entrenar_modelo():
    X, y = cargar_datos()
    print(f"🎧 {len(X)} audios procesados.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_scaled, y)

    # Evaluación
    y_pred = modelo.predict(X_scaled)
    print("\n🧪 Reporte de clasificación:")
    print(classification_report(y, y_pred))

    # Guardar modelo y escalador
    joblib.dump(modelo, "modelo_entrenado.pkl")
    joblib.dump(scaler, "escalador.pkl")
    print("✅ Modelo y escalador guardados.")

if __name__ == "__main__":
    entrenar_modelo()
