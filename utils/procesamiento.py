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

    # PÁNICO
    if energia > 300:
        puntajes['PÁNICO'] += 2
    if proporcion_media > 9:
        puntajes['PÁNICO'] += 3
    if proporcion_altas < 1.3:
        puntajes['PÁNICO'] += 2
    if cero_cruces > 0.13:
        puntajes['PÁNICO'] += 1

    # IRA (incluye furia y euforia)
    if energia > 250:
        puntajes['IRA'] += 2
    if 4 < proporcion_media < 10:
        puntajes['IRA'] += 2
    if proporcion_altas > 1.3:
        puntajes['IRA'] += 2
    if cero_cruces > 0.13:
        puntajes['IRA'] += 2
    if centroide > 3400:
        puntajes['IRA'] += 1

    # TRISTEZA (incluye preocupación)
    if energia < 20:
        puntajes['TRISTEZA'] += 2
    if proporcion_media < 4:
        puntajes['TRISTEZA'] += 2
    if centroide < 3400:
        puntajes['TRISTEZA'] += 2
    if cero_cruces < 0.11:
        puntajes['TRISTEZA'] += 1

    # CALMA (incluye alegría)
    if energia < 15:
        puntajes['CALMA'] += 2
    if proporcion_altas > 1.3 and proporcion_media < 4:
        puntajes['CALMA'] += 3
    if cero_cruces < 0.12:
        puntajes['CALMA'] += 2
    if 3400 < centroide < 4000:
        puntajes['CALMA'] += 1

    return max(puntajes, key=puntajes.get)
