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
    # Basadas en análisis estadístico del dataset
    
    # ---- CALMA (patrones observados) ----
    # Baja energía + proporciones equilibradas
    if energia < 15:
        puntajes['CALMA'] += 2
        
        if 0.7 < prop_altas < 2.5 and 2.0 < prop_media < 4.5:
            puntajes['CALMA'] += 3
            
    # Cero cruces característicos
    if cero_cruces < 0.12:
        puntajes['CALMA'] += 1
        
    # Centroides moderados
    if 2500 < centroide < 4200:
        puntajes['CALMA'] += 1

    # ---- TRISTEZA (patrones observados) ----
    # Energía muy baja + alta proporción de medios
    if energia < 10:
        puntajes['TRISTEZA'] += 2
        
        if prop_media > 2.5:
            puntajes['TRISTEZA'] += 2
            
    # Centroides bajos característicos
    if centroide < 3500:
        puntajes['TRISTEZA'] += 2
        
    # Proporción de altas típica
    if prop_altas < 2.0:
        puntajes['TRISTEZA'] += 1

    # ---- IRA (patrones observados) ----
    # Alta energía es el principal indicador
    if energia > 100:
        puntajes['IRA'] += 4
        
    # Proporciones extremas
    if prop_media > 5.0:
        puntajes['IRA'] += 3
        
    # Cero cruces altos
    if cero_cruces > 0.14:
        puntajes['IRA'] += 2

    # ---- PANICO (patrones observados) ----
    # Energía extremadamente alta
    if energia > 300:
        puntajes['PANICO'] += 4
    elif energia > 100:
        puntajes['PANICO'] += 2
        
    # Proporciones anormales
    if prop_media > 8.0:
        puntajes['PANICO'] += 3
    elif prop_media > 5.0:
        puntajes['PANICO'] += 1
        
    # Combinación única de características
    if cero_cruces > 0.14 and prop_altas < 1.8 and centroide < 3800:
        puntajes['PANICO'] += 2

    # ============= REGLAS DE EXCLUSIÓN =============
    # Basadas en límites observados en el dataset
    
    # Descartar CALMA si energía muy alta
    if energia > 50:
        puntajes['CALMA'] = max(0, puntajes['CALMA'] - 3)
        
    # Descartar TRISTEZA si proporción media muy alta
    if prop_media > 6.0:
        puntajes['TRISTEZA'] = max(0, puntajes['TRISTEZA'] - 2)
        
    # Descartar IRA si energía baja
    if energia < 20:
        puntajes['IRA'] = max(0, puntajes['IRA'] - 3)
        
    # Descartar PANICO si proporción media baja
    if prop_media < 3.0:
        puntajes['PANICO'] = max(0, puntajes['PANICO'] - 3)

    # ============= DESEMPATE INTELIGENTE =============
    max_score = max(puntajes.values())
    if max_score == 0:
        return 'CALMA'  # Valor por defecto
    
    candidatos = [emo for emo, sc in puntajes.items() if sc == max_score]
    
    # Priorizar por energía si hay empate
    if len(candidatos) > 1:
        if energia > 200: 
            if 'IRA' in candidatos: return 'IRA'
            if 'PANICO' in candidatos: return 'PANICO'
        elif energia < 5:
            if 'TRISTEZA' in candidatos: return 'TRISTEZA'
            if 'CALMA' in candidatos: return 'CALMA'
            
        # Prioridad secundaria
        for emo in ['PANICO', 'IRA', 'TRISTEZA', 'CALMA']:
            if emo in candidatos:
                return emo

    return max(puntajes, key=puntajes.get)
