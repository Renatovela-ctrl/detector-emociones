import streamlit as st
import librosa
import matplotlib.pyplot as plt
import numpy as np
from utils.procesamiento import extraer_caracteristicas, modelo, escalador, calcular_fft

st.set_page_config(page_title="Detector de Emociones en la Voz", layout="wide")
st.title("🎙️ Detector de Emociones en la Voz con FFT")

st.markdown("""
### ℹ️ Criterios de Clasificación de Emociones

- **TRISTEZA**: Incluye tristeza y **preocupación**.
- **CALMA**: Serenidad, tranquilidad y también **alegría**.
- **IRA**: Ira, **furia** o incluso **euforia** intensa.
- **PÁNICO**: Estados de ansiedad o **miedo** pronunciado.

> 🔍 El análisis se basa en características espectrales (MFCCs, energía, etc.) extraídas de los **primeros 2 segundos** del audio.
""")

# Sidebar con audios de ejemplo
st.sidebar.header("🔊 Audios de ejemplo")
emociones = {
    "IRA": "ejemplos/ira.wav",
    "PÁNICO": "ejemplos/panico.wav",
    "TRISTEZA": "ejemplos/tristeza.wav",
    "CALMA": "ejemplos/calma.wav"
}

for nombre, path in emociones.items():
    with st.sidebar.expander(f"▶️ {nombre}", expanded=False):
        st.audio(path)

# Carga de audio personalizado
st.header("📤 Análisis de audio personalizado")
audio_file = st.file_uploader("Sube un archivo .wav (voz masculina)", type=["wav"])

if audio_file:
    y, sr = librosa.load(audio_file, sr=None)
    y = y / max(abs(y)) if np.max(np.abs(y)) > 0 else y
    st.audio(audio_file)

    segmento = y[:int(sr * 2)]
    carac = extraer_caracteristicas(segmento, sr)
    X = np.array(carac).reshape(1, -1)
    X = escalador.transform(X)

    # Predicción
    probs = modelo.predict_proba(X)[0]
    clases = modelo.classes_
    emocion = clases[np.argmax(probs)]
    confianza = np.max(probs) * 100

    # Mostrar resultado
    st.subheader(f"🧠 Emoción detectada: **{emocion}**")
    st.markdown(f"📈 Nivel de confianza: **{confianza:.1f}%**")

    # Mostrar probabilidades por clase en gráfico de barras
    st.markdown("#### 🔢 Distribución de probabilidad por emoción")
    st.bar_chart({clase: prob for clase, prob in zip(clases, probs)})

    # Mostrar otras características (no MFCCs)
    st.markdown("#### 📊 Características espectrales (resumen)")
    st.json({
        "Energía": round(carac[13], 5),
        "Cero cruces": round(carac[14], 3),
        "Centroide espectral (Hz)": round(carac[15], 1),
        "Roll-off": round(carac[16], 1),
        "Proporción altas": round(carac[17], 2),
        "Proporción media": round(carac[18], 2),
    })

    # Gráfico FFT
    freqs, magnitudes = calcular_fft(segmento, sr)
    fig, ax = plt.subplots()
    ax.plot(freqs, magnitudes)
    ax.set_title("Transformada de Fourier del segmento")
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("Magnitud")
    st.pyplot(fig)
