import streamlit as st
import librosa
import matplotlib.pyplot as plt
from utils.procesamiento import extraer_caracteristicas, clasificar_emocion

st.set_page_config(page_title="Detector de Emociones en la Voz", layout="wide")
st.title("🎙️ Detector de Emociones en la Voz con FFT")

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
    y = y / max(abs(y))  # Normalización
    st.audio(audio_file)

    # Usar los primeros 2 segundos para análisis
    segmento = y[:int(sr*2)]

    # Análisis espectral
    carac = extraer_caracteristicas(segmento, sr)
    emocion = clasificar_emocion(carac)

    st.subheader(f"🧠 Emoción detectada: **{emocion}**")
    st.json({
        "Energía": round(carac['energia'], 5),
        "Cero cruces": round(carac['cero_cruces'], 3),
        "Centroide espectral (Hz)": round(carac['centroide'], 1),
        "Roll-off": round(carac['rolloff'], 1),
        "Proporción altas": round(carac['proporcion_altas'], 2),
        "Proporción media": round(carac['proporcion_media'], 2),
    })

    # Gráfico FFT
    fig, ax = plt.subplots()
    ax.plot(carac['frecuencias'], carac['magnitudes'])
    ax.set_title("Transformada de Fourier del segmento")
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("Magnitud")
    st.pyplot(fig)