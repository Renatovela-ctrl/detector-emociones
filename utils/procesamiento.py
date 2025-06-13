import numpy as np
import librosa
import joblib
import os

# Cargar modelo y escalador entrenados (solo una vez)
modelo_path = 'modelo_entrenado.pkl'
escalador_path = 'escalador.pkl'

if not os.path.exists(modelo_path) or not os.path.exists(escalador_path):
    raise FileNotFoundError("Faltan archivos del modelo entrenado (.pkl).")

modelo = joblib.load(modelo_path)
escalador = joblib.load(escalador_path)

# -------- Función para calcular características --------
def calcular_fft(segmento, sr):
    N = len(segmento)
    fft = np.fft.fft(segmento)
    magnitud = np.abs(fft)[:N // 2]
    freqs = np.fft.fftfreq(N, 1/sr)[:N // 2]
    return freqs, magnitud

def extraer_caracteristicas(segmento, sr):
    energia = np.sum(segmento**2)
    cero_cruces = librosa.feature.zero_crossing_rate(segmento)[0][0]
    freqs, magnitud = calcular_fft(segmento, sr)
    centroide = np.sum(freqs * magnitud) / (np.sum(magnitud) + 1e-12)
    rolloff = librosa.feature.spectral_rolloff(y=segmento, sr=sr)[0][0]

    bandas = {
        'baja': np.sum(magnitud[(freqs >= 0) & (freqs < 300)]),
        'media': np.sum(magnitud[(freqs >= 300) & (freqs < 1500)]),
        'alta': np.sum(magnitud[(freqs >= 1500)])
    }

    proporcion_altas = bandas['alta'] / (bandas['media'] + 1e-12)
    proporcion_media = bandas['media'] / (bandas['baja'] + 1e-12)

    return [energia, cero_cruces, centroide, rolloff, proporcion_altas, proporcion_media]

# -------- Función para clasificar emoción con modelo --------
def clasificar_emocion(audio_path):
    try:
        segmento, sr = librosa.load(audio_path, sr=None)
        caracteristicas = extraer_caracteristicas(segmento, sr)
        caracteristicas_np = np.array(caracteristicas).reshape(1, -1)
        caracteristicas_np = escalador.transform(caracteristicas_np)
        emocion = modelo.predict(caracteristicas_np)[0]
        return emocion
    except Exception as e:
        return f"Error en clasificación: {str(e)}"
