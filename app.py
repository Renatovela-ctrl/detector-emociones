import streamlit as st
import librosa
import matplotlib.pyplot as plt
import numpy as np
from utils.procesamiento import extraer_caracteristicas, modelo, escalador

st.set_page_config(page_title="Detector de Emociones en la Voz", layout="wide")
st.title("🎙️ Detector de Emociones en la Voz con FFT + MFCC")

st.markdown("""
### ℹ️ Criterios de Clasificación de Emociones

- **TRISTEZA**: Incluye tanto emociones de tristeza como **preocupación**.
- **CALMA**: Representa estados de calma, serenidad y también incluye **alegría**.
- **IRA**: Abarca la ira y sus variantes intensas como la **furia** y también **euforia**.
- **PÁNICO**: Corresponde a estados de ansiedad o pánico pronunciado.

> 🔍 La clasificación se basa en análisis de características espectrales (MFCCs y otras) extraídas de los primeros 2 segundos del audio.
""")

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

st.header("📤 Análisis de audio personalizado")
audio_file = st.file_uploader("Sube un archivo .wav (voz masculina)", type=["wav"])

if audio_file:
    y, sr = librosa.load(audio_file, sr=None)
    y = y / max(abs(y)) if np.max(np.abs(y)) > 0 else y
    st.audio(audio_file)

    segmento = y[:int(sr*2)]

    carac = extraer_caracteristicas(segmento, sr)
    X = np.array(carac).reshape(1, -1)
    X = escalador.transform(X)
    emocion = modelo.predict(X)[0]

    st.subheader(f"🧠 Emoción detectada: **{emocion}**")

    # Mostrar MFCCs
    st.markdown("#### 🎚️ Coeficientes MFCC")
    for i in range(13):
        st.write(f"MFCC {i+1}: {round(carac[i], 2)}")

    # Mostrar otras características
    st.markdown("#### 📊 Otras características espectrales")
    st.json({
        "Energía": round(carac[13], 5),
        "Cero cruces": round(carac[14], 3),
        "Centroide espectral (Hz)": round(carac[15], 1),
        "Roll-off": round(carac[16], 1),
        "Proporción altas": round(carac[17], 2),
        "Proporción media": round(carac[18], 2),
    })

    # Gráfico FFT
    from utils.procesamiento import calcular_fft
    freqs, magnitudes = calcular_fft(segmento, sr)
    fig, ax = plt.subplots()
    ax.plot(freqs, magnitudes)
    ax.set_title("Transformada de Fourier del segmento")
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("Magnitud")
    st.pyplot(fig)
