import streamlit as st
import requests
import cv2
import numpy as np
import pandas as pd
import time
from datetime import datetime

# =============================
# Configuration
# =============================
st.set_page_config(page_title="♻️ Smart Waste Segregation System", layout="wide")
API_URL = "http://127.0.0.1:8000/imageprocess"

# Initialize history
if "history" not in st.session_state:
    st.session_state.history = []

# =============================
# Utility Functions
# =============================
def map_category(predicted_class):
    biodegradable = ["cardboard", "paper"]
    recyclable = ["glass", "metal", "plastic"]
    non_biodegradable = ["trash"]

    if predicted_class in biodegradable:
        return "Biodegradable"
    elif predicted_class in recyclable:
        return "Recyclable"
    else:
        return "Non-Biodegradable"

def get_ai_recommendation(category):
    if category == "Biodegradable":
        return "🌿 Compost this waste to enrich soil and reduce landfill burden."
    elif category == "Recyclable":
        return "♻️ Recycle this material to conserve resources and energy."
    else:
        return "🚯 Dispose safely in non-recyclable waste bins to prevent pollution."

# =============================
# Sidebar
# =============================
st.sidebar.title("⚙️ Navigation")
menu = st.sidebar.radio(
    "Choose Mode",
    ["📊 Dashboard", "📁 Upload Image", "🎥 Real-time Detection"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Developed by Madhav Rao | AI for a Cleaner Planet 🌏")

# =============================
# Dashboard View
# =============================
if menu == "📊 Dashboard":
    st.title("📈 Waste Classification Dashboard")
    st.markdown("View statistics, analytics, and system insights.")

    if len(st.session_state.history) == 0:
        st.info("No data yet. Upload or run live detection.")
    else:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(df["Mapped_Category"].value_counts())
        with col2:
            st.line_chart(df["Confidence"])

        st.markdown("### 🌍 Environmental Impact Summary")
        total = len(df)
        bio = len(df[df["Mapped_Category"] == "Biodegradable"])
        rec = len(df[df["Mapped_Category"] == "Recyclable"])
        nonb = len(df[df["Mapped_Category"] == "Non-Biodegradable"])

        st.metric("Total Processed", total)
        st.progress((bio + rec) / total)

        st.success(f"Biodegradable: {bio} | Recyclable: {rec} | Non-Biodegradable: {nonb}")

# =============================
# Upload Image Mode
# =============================
elif menu == "📁 Upload Image":
    st.title("📸 Upload Waste Image for Classification")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", width=300)
        if st.button("🔍 Classify Waste"):
            files = {"image": uploaded_file.getvalue()}
            response = requests.post(API_URL, files=files)
            if response.ok:
                result = response.json()["message"]
                predicted_class, acc = result.split(", ")
                category = map_category(predicted_class)
                rec = get_ai_recommendation(category)

                st.success(f"**Predicted Class:** {predicted_class}")
                st.info(f"**Mapped Category:** {category}")
                st.progress(float(acc.strip('%')) / 100)
                st.write(f"**Accuracy:** {acc}")
                st.markdown(f"### 💡 AI Recommendation:\n{rec}")

                # Log to history
                st.session_state.history.append({
                    "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Predicted_Class": predicted_class,
                    "Mapped_Category": category,
                    "Confidence": float(acc.strip('%'))
                })

# =============================
# Real-time Detection Mode (OpenCV)
# =============================
elif menu == "🎥 Real-time Detection":
    st.title("🎥 Live Waste Detection via Camera (AI-Powered)")
    st.markdown("AI continuously detects waste in real-time using your webcam.")

    run = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])

    cap = None
    if run:
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not accessible.")
                break

            # Resize and encode
            _, buffer = cv2.imencode('.jpg', frame)
            files = {"image": buffer.tobytes()}

            try:
                response = requests.post(API_URL, files=files)
                if response.ok:
                    result = response.json()["message"]
                    predicted_class, acc = result.split(", ")
                    category = map_category(predicted_class)
                    rec = get_ai_recommendation(category)

                    # Display info on frame
                    text = f"{predicted_class} ({acc}) - {category}"
                    cv2.putText(frame, text, (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    st.session_state.history.append({
                        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Predicted_Class": predicted_class,
                        "Mapped_Category": category,
                        "Confidence": float(acc.strip('%'))
                    })

            except Exception as e:
                st.error(f"Error: {e}")

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            time.sleep(0.5)

        cap.release()
        st.success("✅ Camera stopped")

