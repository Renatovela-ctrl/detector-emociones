import numpy as np
import librosa

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

    return {
        'energia': energia,
        'centroide': centroide,
        'rolloff': rolloff,
        'cero_cruces': cero_cruces,
        'proporcion_altas': proporcion_altas,
        'proporcion_media': proporcion_media,
        'frecuencias': freqs,
        'magnitudes': magnitud
    }

def clasificar_emocion(carac):
    energia = carac['energia']
    cero_cruces = carac['cero_cruces']
    centroide = carac['centroide']
    proporcion_altas = carac['proporcion_altas']
    proporcion_media = carac['proporcion_media']

    puntajes = {'IRA': 0, 'PÁNICO': 0, 'TRISTEZA': 0, 'CALMA': 0}

    if energia > 1500:
        puntajes['CALMA'] += 2

    if cero_cruces > 0.03:
        puntajes['IRA'] += 2
    elif cero_cruces > 0.01:
        puntajes['PÁNICO'] += 1
    elif cero_cruces < 0.01:
        puntajes['TRISTEZA'] += 1
        puntajes['CALMA'] += 1

    if centroide > 3300:
        puntajes['CALMA'] += 2
    elif centroide < 2800:
        puntajes['IRA'] += 1
        puntajes['PÁNICO'] += 1

    if proporcion_altas > 1.1:
        puntajes['CALMA'] += 1
    elif proporcion_altas < 0.9:
        puntajes['PÁNICO'] += 2

    if proporcion_media > 5:
        puntajes['PÁNICO'] += 2
    elif proporcion_media < 4:
        puntajes['TRISTEZA'] += 1

    return max(puntajes, key=puntajes.get)
