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

    # --- DESCARTES preventivos ---
    if energia < 50:
        puntajes['IRA'] = 0
    if energia < 80:
        puntajes['PÁNICO'] -= 2  # baja energía ≠ pánico
    if proporcion_media < 4:
        puntajes['PÁNICO'] -= 1  # media muy baja ≠ pánico
    if energia < 15 and proporcion_media < 3 and centroide > 3300:
        puntajes['CALMA'] += 3  # énfasis a calma tranquila

    # --- PÁNICO ---
    if energia > 300 and proporcion_media > 9 and proporcion_altas < 1.3:
        puntajes['PÁNICO'] += 4
    if cero_cruces > 0.13:
        puntajes['PÁNICO'] += 1
    if centroide < 3400:
        puntajes['PÁNICO'] += 1

    # Penalización adicional si calma parece pánico
    if energia < 30 and proporcion_altas > 1.3 and proporcion_media < 5:
        puntajes['PÁNICO'] -= 2
        puntajes['CALMA'] += 2

    # --- IRA ---
    if energia > 250 and 4 < proporcion_media < 10 and proporcion_altas > 1.3:
        puntajes['IRA'] += 4
    if cero_cruces > 0.13:
        puntajes['IRA'] += 2
    if centroide > 3400:
        puntajes['IRA'] += 2
    if proporcion_media < 3:
        puntajes['IRA'] -= 1

    # --- TRISTEZA ---
    if energia < 20 and proporcion_media < 4 and centroide < 3400:
        puntajes['TRISTEZA'] += 4
    if cero_cruces < 0.11:
        puntajes['TRISTEZA'] += 2
    if proporcion_altas < 1.4:
        puntajes['TRISTEZA'] += 1
    if energia > 100:
        puntajes['TRISTEZA'] -= 1

    # --- CALMA ---
    if energia < 15 and proporcion_altas > 1.3 and proporcion_media < 4:
        puntajes['CALMA'] += 4
    if cero_cruces < 0.12:
        puntajes['CALMA'] += 2
    if 3400 < centroide < 4000:
        puntajes['CALMA'] += 2
    if energia < 20 and proporcion_media < 3 and proporcion_altas > 1.4:
        puntajes['CALMA'] += 2

    # --- Desempates con prioridad emocional ---
    if len(set(puntajes.values())) != len(puntajes.values()):
        prioridad = ['IRA', 'PÁNICO', 'TRISTEZA', 'CALMA']
        maximo = max(puntajes.values())
        for emo in prioridad:
            if puntajes[emo] == maximo:
                return emo

    return max(puntajes, key=puntajes.get)
