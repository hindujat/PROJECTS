import cv2
import numpy as np
import os
import random
import pygame
import streamlit as st
import time
from keras.models import load_model

# Initialize Pygame mixer
pygame.mixer.init()

# Load trained emotion detection model
model = load_model('emotion_model.h5')

# Emotion categories
emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Available languages
languages = ['English', 'Tamil', 'Hindi']

# Base directory for songs
BASE_DIR = r"C:\Users\rishi\Downloads\music-recommendation-main\dataset\songs"

# Function to get song path based on emotion and language
def get_song_path(emotion, language):
    song_folder = os.path.join(BASE_DIR, language.lower(), emotion)
    if os.path.exists(song_folder):
        songs = [f for f in os.listdir(song_folder) if f.lower().endswith(('.mp3', '.wav'))]
        if songs:
            return os.path.join(song_folder, random.choice(songs))
    return None

# Cooldown timer settings (5 minutes)
COOLDOWN_TIME = 4 * 60
last_played_time = 0
current_song_path = None

# Function to play a song with cooldown
def play_song_with_cooldown(emotion, language):
    global last_played_time, current_song_path
    current_time = time.time()
    if current_time - last_played_time < COOLDOWN_TIME and current_song_path:
        return
    song_path = get_song_path(emotion, language)
    if song_path:
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
        pygame.mixer.music.load(song_path)
        pygame.mixer.music.play()
        st.success(f"🎵 Now Playing: {os.path.basename(song_path)} ({language})")
        current_song_path = song_path
        last_played_time = current_time
    else:
        st.warning(f"⚠ No song found for {emotion} in {language}. Check your folder structure.")

# Apply CSS styling
st.markdown("""
    <style>
    .stApp { background: linear-gradient(145deg, #1f2933, #3b4a5a); color: #e1e1e1; }
    h1 { text-shadow: 2px 2px 10px #00ffff; }
    .stButton>button { background: rgba(0, 0, 0, 0.5); color: #00ffff; border-radius: 10px; }
    .stButton>button:hover { background: #00ffff; color: #141E30; transform: scale(1.05); }
    </style>
    """, unsafe_allow_html=True)

# Streamlit UI setup
st.title("🎵 Music Recommendations Based on Facial Recognition")
st.write("Detects your facial emotion and plays a song based on it!")

# Language selection
selected_language = st.selectbox("🌍 Choose Language:", languages)

# Video capture setup
cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()
emotion_text = st.empty()

# Button to start detection
hidden_play_button = st.button("🎶 Detect Emotion & Play Music")

# ⏹ Stop Music Button
if st.button("⏹ Stop Music"):
    try:
        if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
            st.info("🛑 Music stopped.")
        else:
            st.warning("⚠ No music is currently playing.")
    except Exception as e:
        st.error(f"⚠ Error stopping music: {e}")


# Track last detected emotion
last_detected_emotion = None

# Start emotion detection and song playing
if hidden_play_button:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("⚠ Camera not working.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        detected_emotion = "Neutral"
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = np.expand_dims(face, axis=0).reshape(1, 48, 48, 1) / 255.0
            prediction = model.predict(face)
            top_emotion_index = np.argmax(prediction)
            detected_emotion = emotion_classes[top_emotion_index]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f'Emotion: {detected_emotion}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        emotion_text.write(f"😃 Detected Emotion: {detected_emotion}")
        play_song_with_cooldown(detected_emotion, selected_language)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()