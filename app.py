import streamlit as st
import numpy as np
import librosa
import joblib
from transformers import Wav2Vec2FeatureExtractor, HubertModel
import torch

# --- CONFIG ---
MODEL_PATH = "hubert_accent_model_full.pkl" # The file you downloaded earlier
HUBERT_ID = "facebook/hubert-base-ls960"

# --- UI LAYOUT ---
st.set_page_config(page_title="Accent-Aware Dining", page_icon="🍛")

st.title("🍛 Accent-Aware Cuisine Recommender")
st.write("Speak your order! We'll detect your region and suggest dishes.")

# --- CUISINE DATABASE ---
menu = {
    0: {"Region": "Andhra Pradesh", "Dishes": "Spicy Hyderabadi Biryani, Gongura Pachadi, Pesarattu"},
    1: {"Region": "Gujarat", "Dishes": "Dhokla, Thepla, Undhiyu (Sweet & Savory)"},
    2: {"Region": "Jharkhand/Hindi", "Dishes": "Litti Chokha, Sattu Paratha"},
    3: {"Region": "Karnataka", "Dishes": "Bisi Bele Bath, Mysore Masala Dosa"},
    4: {"Region": "Kerala", "Dishes": "Appam with Stew, Puttu & Kadala Curry, Karimeen Fry"},
    5: {"Region": "Tamil Nadu", "Dishes": "Idli Sambar, Chicken Chettinad, Pongal"}
}

# --- BACKEND (Cached) ---
@st.cache_resource
def load_assets():
    checkpoint = joblib.load(MODEL_PATH)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(HUBERT_ID)
    model = HubertModel.from_pretrained(HUBERT_ID)
    return checkpoint['model'], checkpoint['scaler'], processor, model

try:
    clf, scaler, processor, model = load_assets()
    st.success("System Ready: AI Model Loaded.")
except:
    st.error("Model file not found. Please upload 'hubert_accent_model_full.pkl'")

# --- AUDIO INPUT ---
audio_file = st.file_uploader("Upload Audio (WAV/MP3)", type=['wav', 'mp3'])

if audio_file is not None:
    st.audio(audio_file)
    
    if st.button("Analyze Accent"):
        with st.spinner("Analyzing speech patterns..."):
            # 1. Load and Resample
            audio, sr = librosa.load(audio_file, sr=16000)
            
            # 2. Extract Features (HuBERT)
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Mean pool last layer
            embedding = torch.mean(outputs.last_hidden_state, dim=1).numpy()
            
            # 3. Predict
            embedding_scaled = scaler.transform(embedding)
            prediction = clf.predict(embedding_scaled)[0]
            
            # 4. Show Results
            result = menu.get(prediction, {"Region": "Unknown", "Dishes": "General Indian Menu"})
            
            st.divider()
            st.subheader(f"🏳️ Detected Accent: {result['Region']}")
            st.info(f"🍽️ **Recommended for you:** {result['Dishes']}")
