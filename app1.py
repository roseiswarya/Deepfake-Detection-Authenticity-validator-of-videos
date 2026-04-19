import streamlit as st
import sqlite3
import torch
import torch.nn as nn
import numpy as np
from deepface import DeepFace
import cv2
import random
import tempfile
import os
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import PIL.Image
import google.generativeai as genai # Placeholder for backwards compatibility if needed, but 
from xai_utils import generate_occlusion_map, generate_forensic_report
from xai_utils import generate_occlusion_map, generate_forensic_report

# -------------------- DB Setup --------------------
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)''')
conn.commit()

def add_user(username, password):
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def verify_user(username, password):
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    return c.fetchone() is not None


# -------------------- EMAIL ALERT FUNCTION --------------------
def send_email_alert(username, probability):
    sender_email = "dummyemailsender24@gmail.com"  # 🔴 CHANGE
    receiver_email = "vanidurai2004@gmail.com"  # 🔴 CHANGE
    app_password = "mbijbmtnndzvmxaw"  # 🔴 CHANGE

    subject = "🚨 Deepfake Alert Detected!"
    body = f"""
Deepfake Video Detected!

User: {username}
Fake Probability: {round(probability, 2)}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Please review immediately.
"""

    try:
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = receiver_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, app_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()

        print("Email alert sent successfully!")

    except Exception as e:
        print("Failed to send email:", e)


# -------------------- Model Setup --------------------
class DeepfakeClassifier(nn.Module):
    def __init__(self):
        super(DeepfakeClassifier, self).__init__()
        self.efficientnet = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.efficientnet(x)
        x = self.fc(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepfakeClassifier().to(device)
model.load_state_dict(torch.load("efficientnetv2_fakeface_classifier.pth", map_location=device))
model.eval()


def extract_features_from_video(video_path):
    def extract_frames(video_path, frame_rate=5):
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_rate == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_count += 1
        cap.release()
        return frames

    frames = extract_frames(video_path, frame_rate=2)
    if len(frames) > 100:
        frames = random.sample(frames, 100)

    embeddings = []
    for frame in frames:
        features = DeepFace.represent(frame, model_name='Facenet512', enforce_detection=False)
        if features:
            embeddings.append(features[0]['embedding'])

    return np.mean(embeddings, axis=0) if embeddings else None


# -------------------- Local XAI Helper --------------------
def get_classifier_input(frame):
    """
    Helper for XAI: Converts a single frame into the 512-embedding expected by the classifier.
    """
    try:
        # Detect face and get embedding
        features = DeepFace.represent(frame, model_name='Facenet512', enforce_detection=False)
        if features:
            embedding = features[0]['embedding']
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)
    except Exception as e:
        return None
    return None


# -------------------- Session Init --------------------
if "auth" not in st.session_state:
    st.session_state.auth = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "page" not in st.session_state:
    st.session_state.page = "home"


# -------------------- Auth Pages --------------------
def show_homepage():
    st.set_page_config(page_title="Deepfake Detection App", layout="centered")
    st.title("🎭 Deepfake Video Detection")
    st.image("Synthetic-Humans-The-Digtal-Speaker.jpg", use_column_width=True)
    st.subheader("Detect deepfakes with AI. Please sign in or sign up to get started.")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if st.button("🔐 Sign In"):
            st.session_state.page = "signin"
            st.rerun()

    with col5:
        if st.button("🆕 Sign Up"):
            st.session_state.page = "signup"
            st.rerun()


def signin_page():
    st.title("🔐 Sign In")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if verify_user(username, password):
            st.session_state.auth = True
            st.session_state.username = username
            st.session_state.page = "main"
            st.rerun()
        else:
            st.error("Invalid username or password")

    if st.button("Don't have an account? Sign Up"):
        st.session_state.page = "signup"
        st.rerun()


def signup_page():
    st.title("🆕 Sign Up")
    username = st.text_input("Choose a Username")
    password = st.text_input("Choose a Password", type="password")

    if st.button("Create Account"):
        if add_user(username, password):
            st.success("Account created! Please sign in.")
            st.session_state.page = "signin"
            st.rerun()
        else:
            st.error("Username already exists!")

    if st.button("Already have an account? Sign In"):
        st.session_state.page = "signin"
        st.rerun()


# -------------------- Main App --------------------
def main_app():
    st.title(f"Welcome, {st.session_state.username} 👋")
    st.subheader("Upload a video to detect if it's real or deepfake")

    with st.sidebar:
        st.title("⚙️ Settings")
        st.info("Local Forensic AI is active. No internet connection is required for detection or explanation.")

    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_file.write(uploaded_file.getbuffer())
        video_path = temp_file.name

        st.video(video_path)
        st.info("Analyzing video... Please wait.")

        features = extract_features_from_video(video_path)

        if features is not None:
            input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)[0].cpu().numpy()
                predicted_class = np.argmax(probabilities)

            if predicted_class == 0:
                st.success("✅ **Real** video")
                st.write("Probability of being real:", round(probabilities[0], 2))
            else:
                st.error("❌ **Fake** video")
                st.write("Probability of being fake:", round(probabilities[1], 2))

            # --- Local Forensic Deep-Dive Section ---
            st.markdown("---")
            st.subheader("🔍 Forensic Deep-Dive (Local XAI)")
            
            result_text = "Real" if predicted_class == 0 else "Fake"
            confidence = float(probabilities[predicted_class])
            
            with st.status("Analyzing forensic artifacts locally..."):
                # 1. Get sample frame
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
                success, sample_frame = cap.read()
                cap.release()

                if success:
                    # Generate the heatmap
                    # Optimization: Stride increased and face cropping handled inside xai_utils
                    heatmap_img = generate_occlusion_map(model, sample_frame)
                    
                    if heatmap_img is not None:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB), caption="Sample Frame")
                        with col2:
                            st.image(cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB), caption="Forensic Heatmap (Red = Suspected Manipulation)")
                
                    # 2. Local Forensic Report
                    st.write("### 📄 Technical Report")
                    forensic_report = generate_forensic_report(confidence, result_text)
                    st.info(forensic_report)
                else:
                    st.warning("Could not capture frame for forensic analysis.")

            if predicted_class == 1:
                # 🚨 SEND EMAIL ALERT
                send_email_alert(st.session_state.username, probabilities[1])

                st.warning("📧 Email alert sent!")

        else:
            st.warning("Could not extract features from the video.")

        try:
            temp_file.close()
            time.sleep(2)
            os.remove(video_path)
        except Exception as e:
            st.warning(f"Could not delete temp file: {e}")

    if st.button("Logout"):
        st.session_state.auth = False
        st.session_state.page = "home"
        st.rerun()


# -------------------- Page Routing --------------------
if not st.session_state.auth:
    if st.session_state.page == "home":
        show_homepage()
    elif st.session_state.page == "signin":
        signin_page()
    elif st.session_state.page == "signup":
        signup_page()
else:
    main_app()
