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

    puntajes = {'CALMA': 0, 'TRISTEZA': 0, 'IRA': 0, 'PANICO': 0}

    # ============= REGLAS MEJORADAS =============
    # Nueva regla fuerte para CALMA con cero cruces muy bajos
    if cero_cruces < 0.05:
        puntajes['CALMA'] += 8  # Puntuación muy alta
        # Refuerzo adicional si cumple características típicas de calma
        if 2500 < centroide < 4200 and 0.5 < prop_altas < 2.0 and 2.0 < prop_media < 4.0:
            puntajes['CALMA'] += 4

    # Regla de exclusión fuerte contra IRA cuando cero cruces es muy bajo
    if cero_cruces < 0.05:
        puntajes['IRA'] -= 10  # Descuento masivo
        puntajes['PANICO'] -= 8

    # ---- CALMA (ajustes específicos) ----
    if energia < 50:
        puntajes['CALMA'] += 2
        
        if 0.5 < prop_altas < 2.5 and 2.0 < prop_media < 4.5:
            puntajes['CALMA'] += 3
            
    if cero_cruces < 0.12:
        puntajes['CALMA'] += 2
        
    if 2500 < centroide < 4200:
        puntajes['CALMA'] += 2

    # ---- TRISTEZA (sin cambios mayores) ----
    if energia < 10:
        puntajes['TRISTEZA'] += 2
        
    if centroide < 3500:
        puntajes['TRISTEZA'] += 2
        
    if prop_altas < 2.0:
        puntajes['TRISTEZA'] += 1

    # ---- IRA (protecciones adicionales) ----
    # Requerir cero cruces altos para puntuar como IRA
    if energia > 100 and cero_cruces > 0.1:
        puntajes['IRA'] += 4
        
    if prop_media > 5.0 and cero_cruces > 0.1:
        puntajes['IRA'] += 3
        
    if cero_cruces > 0.14:
        puntajes['IRA'] += 2

    # ---- PANICO (sin cambios mayores) ----
    if energia > 300:
        puntajes['PANICO'] += 4
        
    if prop_media > 8.0:
        puntajes['PANICO'] += 3

    # ============= REGLAS DE EXCLUSIÓN =============
    # Exclusión fuerte de IRA para audios con cero cruces bajos
    if cero_cruces < 0.08:
        puntajes['IRA'] = max(0, puntajes['IRA'] - 6)
        
    if prop_media < 3.0:
        puntajes['PANICO'] = max(0, puntajes['PANICO'] - 3)

    # ============= DESEMPATE INTELIGENTE =============
    max_score = max(puntajes.values())
    if max_score == 0:
        # Si todo es cero, priorizar CALMA cuando cero_cruces es bajo
        if cero_cruces < 0.1:
            return 'CALMA'
        return 'TRISTEZA'  # Valor por defecto conservador
    
    candidatos = [emo for emo, sc in puntajes.items() if sc == max_score]
    
    # Priorizar CALMA cuando los cero cruces son muy bajos
    if len(candidatos) > 1 and cero_cruces < 0.05:
        return 'CALMA'
            
    return max(puntajes, key=puntajes.get)
