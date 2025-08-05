import pandas as pd
import streamlit as st
import joblib
import base64

# Set page config
st.set_page_config(page_title="🎶 Streamify Song Vibe Classifier", layout="wide")

# Background Image Setup
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded_image}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}
        </style>
    """, unsafe_allow_html=True)

# Load background image
set_background("music.jpeg")  # Make sure this image exists in your folder

# Global styling tweaks
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        .stApp {
            background-color: transparent !important;
        }

        .block-container {
            padding: 0rem 2rem 2rem;
            max-width: 100% !important;
            width: 100%;
            
            border-radius: 12px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }

        h1, h3, .stSlider label {
             color: #fff !important;
            text-shadow: 1px 1px 2px rgba(255,255,255,0.5);
            
        }

        .stButton > button {
            background-color: #8e44ad;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px 20px;
        }

        .stButton > button:hover {
            background-color: #732d91;
        }

        .stSlider > div {
            background: rgba(255, 255, 255, 0.8);
            padding: 10px;
            border-radius: 10px;
        }

        .stSuccess {
            background-color: rgba(0, 0, 0, 0.4) !important;
            color: #fff !important;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center;'>🎵 Streamify: Song Vibe Classifier</h1>", unsafe_allow_html=True)
st.markdown("### 🔊 Tune in your song's features and discover its vibe!")

# Load ML models
model = joblib.load('spotify_kmean_model.pkl')
pca = joblib.load('spotify_pca.pkl')
scaler = joblib.load('spotify_scaler.pkl')

# Input sliders
col1, col2 = st.columns(2)
with col1:
    popularity = st.slider('🎯 Popularity', 0, 100, 50)
    duration_ms = st.slider('⏱️ Duration (ms)', 43235, 392638, 200000)
    explicit = st.slider('🔞 Explicit (0=No, 1=Yes)', 0, 1, 0)
    danceability = st.slider('🕺 Danceability', 0.06, 0.99, 0.5)
    energy = st.slider('⚡ Energy', 0.01, 1.0, 0.5)
    key = st.slider('🎼 Music Key (0=C, 11=B)', 0, 11, 5)
    loudness = st.slider('🔊 Loudness (dB)', -17.9, 1.7, -6.0)

with col2:
    mode = st.slider('🎶 Mode (0=Minor, 1=Major)', 0, 1, 1)
    speechiness = st.slider('🗣️ Speechiness', 0.0, 1.0, 0.1)
    acousticness = st.slider('🎸 Acousticness', 0.0, 1.0, 0.3)
    instrumentalness = st.slider('🎻 Instrumentalness', 0.0, 0.1, 0.01)
    liveness = st.slider('🎤 Liveness', 0.009, 0.534, 0.1)
    valence = st.slider('😊 Valence (Mood)', 0.0, 0.994, 0.5)
    tempo = st.slider('🎵 Tempo (BPM)', 36.542, 209.143, 120.0)
    time_signature = st.slider('🕐 Time Signature', 0, 5, 4)

# Predict button
if st.button("🔍 Predict Song Group"):
    input_df = pd.DataFrame({
        'popularity': [popularity],
        'duration_ms': [duration_ms],
        'explicit': [explicit],
        'danceability': [danceability],
        'energy': [energy],
        'key': [key],
        'loudness': [loudness],
        'mode': [mode],
        'speechiness': [speechiness],
        'acousticness': [acousticness],
        'instrumentalness': [instrumentalness],
        'liveness': [liveness],
        'valence': [valence],
        'tempo': [tempo],
        'time_signature': [time_signature]
    })

    # Apply model transformations
    scaled = scaler.transform(input_df)
    reduced = pca.transform(scaled)
    cluster = model.predict(reduced)[0]

    # Cluster label mapping
    label_map = {
        0: "🎧 Chill Pop / Acoustic Groove",
        1: "💃 Mainstream Dance / EDM-Pop",
        2: "🎤 Stripped Down / Acoustic Ballads"
    }
    predicted_label = label_map.get(cluster, f"Unknown Cluster {cluster}")

    # Display result
    st.markdown("---")
    st.success(f"🎼 **Predicted Vibe**: **{predicted_label}**")
