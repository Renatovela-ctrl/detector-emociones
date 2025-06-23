import numpy as np
import librosa
import joblib
import os

modelo_path = 'modelo_entrenado.pkl'
escalador_path = 'escalador.pkl'

if not os.path.exists(modelo_path) or not os.path.exists(escalador_path):
    raise FileNotFoundError("Faltan archivos del modelo entrenado (.pkl).")

modelo = joblib.load(modelo_path)
escalador = joblib.load(escalador_path)

def calcular_fft(segmento, sr):
    N = len(segmento)
    fft = np.fft.fft(segmento)
    magnitud = np.abs(fft)[:N // 2]
    freqs = np.fft.fftfreq(N, 1/sr)[:N // 2]
    return freqs, magnitud

def extraer_caracteristicas(segmento, sr):
    if len(segmento) < sr * 2:
        segmento = np.pad(segmento, (0, sr*2 - len(segmento)))
    else:
        segmento = segmento[:sr*2]

    segmento = segmento / np.max(np.abs(segmento)) if np.max(np.abs(segmento)) > 0 else segmento
    freqs, magnitud = calcular_fft(segmento, sr)

    energia = np.sum(segmento ** 2)
    cero_cruces = librosa.feature.zero_crossing_rate(segmento)[0].mean()
    centroide = librosa.feature.spectral_centroid(y=segmento, sr=sr)[0].mean()
    rolloff = librosa.feature.spectral_rolloff(y=segmento, sr=sr)[0].mean()

    bandas = {
        'baja': np.sum(magnitud[(freqs >= 0) & (freqs < 300)]),
        'media': np.sum(magnitud[(freqs >= 300) & (freqs < 1500)]),
        'alta': np.sum(magnitud[(freqs >= 1500)])
    }

    proporcion_altas = bandas['alta'] / (bandas['media'] + 1e-6)
    proporcion_media = bandas['media'] / (bandas['baja'] + 1e-6)

    mfccs = librosa.feature.mfcc(y=segmento, sr=sr, n_mfcc=13)
    mfccs_mean = mfccs.mean(axis=1)

    return list(mfccs_mean) + [
        energia,
        cero_cruces,
        centroide,
        rolloff,
        proporcion_altas,
        proporcion_media
    ]
