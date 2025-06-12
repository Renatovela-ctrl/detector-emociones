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
    energia       = carac['energia']
    cero_cruces   = carac['cero_cruces']
    centroide     = carac['centroide']
    prop_altas    = carac['proporcion_altas']
    prop_media    = carac['proporcion_media']

    puntajes = {'CALMA': 0, 'TRISTEZA': 0, 'IRA': 0, 'PÁNICO': 0}

    # —————— REGLAS DE EXCLUSIÓN INICIAL ——————
    # Si cero cruces = 0, fuertemente CALMA
    if cero_cruces == 0:
        puntajes['CALMA'] += 5
        puntajes['IRA']    -= 3
        puntajes['PÁNICO'] -= 3

    # Si energía < 20 → descartar IRA/PÁNICO
    if energia < 20:
        puntajes['IRA']    = max(puntajes['IRA'] - 3, 0)
        puntajes['PÁNICO'] = max(puntajes['PÁNICO'] - 3, 0)

    # Si proporción media < 3 → descartar PÁNICO
    if prop_media < 3:
        puntajes['PÁNICO'] = max(puntajes['PÁNICO'] - 2, 0)

    # —————— REGLAS PRINCIPALES ——————

    # CALMA (incluye alegría)
    if energia < 15 and prop_altas > 1.3 and prop_media < 4:
        puntajes['CALMA'] += 4
    if cero_cruces < 0.12 and 3000 < centroide < 4000:
        puntajes['CALMA'] += 2
    if energia < 50 and cero_cruces < 0.05:
        puntajes['CALMA'] += 3

    # TRISTEZA (incluye preocupación)
    if energia < 50 and prop_media < 4 and centroide < 3400:
        puntajes['TRISTEZA'] += 4
    if cero_cruces < 0.11:
        puntajes['TRISTEZA'] += 2
    if prop_altas < 1.4:
        puntajes['TRISTEZA'] += 1
    if energia > 200:
        puntajes['TRISTEZA'] -= 2  # contradicción con tristeza

    # IRA (incluye furia y euforia)
    if energia > 250 and prop_media > 4 and prop_altas > 1.3:
        puntajes['IRA'] += 4
    if cero_cruces > 0.13 and centroide > 3400:
        puntajes['IRA'] += 2
    if prop_media < 3:
        puntajes['IRA'] -= 1  # incoherente con ira

    # PÁNICO
    if energia > 300 and prop_media > 9 and prop_altas < 1.3:
        puntajes['PÁNICO'] += 4
    if cero_cruces > 0.13 and centroide < 3300:
        puntajes['PÁNICO'] += 2
    if prop_altas < 1.0:
        puntajes['PÁNICO'] += 1
    if prop_media < 5:
        puntajes['PÁNICO'] -= 2  # desacredita pánico leve

    # —————— DESEMPATE INTELIGENTE ——————
    # En caso de empate, prioridad a:
    # CALMA > TRISTEZA > IRA > PÁNICO (voz monótona vs. intensa)
    max_score = max(puntajes.values())
    candidatos = [emo for emo, sc in puntajes.items() if sc == max_score]
    if len(candidatos) > 1:
        for emo in ['CALMA','TRISTEZA','IRA','PÁNICO']:
            if emo in candidatos:
                return emo

    return max(puntajes, key=puntajes.get)

