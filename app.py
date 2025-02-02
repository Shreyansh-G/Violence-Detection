import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import tempfile
import os
import matplotlib.pyplot as plt

# Disable GPU usage
tf.config.set_visible_devices([], 'GPU')

# Load trained model
model = tf.keras.models.load_model("movileV3_89.keras")

# Function to preprocess an image
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to process video and extract frames
def extract_frames(video_path, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:  # Extract every Nth frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)  # Convert to PIL image
            frames.append(preprocess_image(frame))  # Preprocess for model

        frame_count += 1

    cap.release()
    return frames

# Function to predict image class
def predict_image(image):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)[0][0]
    return "Violence" if prediction > 0.5 else "Non-Violence", float(prediction)  # Convert float32 to float

# Streamlit UI Design
st.title("🛡️ Violence Detection System")
st.write("Upload an **image** or **video** to detect violence automatically.")

# 🎨 Example Images for Better UI
st.subheader("📌 Example Images")
example_images = ["example1.jpg","example2.jpg"]  # Replace with actual paths

cols = st.columns(len(example_images))
for idx, img_path in enumerate(example_images):
    with cols[idx]:
        st.image(img_path, use_container_width=True)

# 🖼️ Image Upload Section
st.subheader("📸 Upload an Image")
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    
    # Split into two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        label, confidence = predict_image(image)
        st.markdown(f"### Prediction: **{label}**")
        st.progress(min(max(confidence, 0), 1))  # Ensure confidence is in valid range

        if label == "Violence":
            st.error(f"⚠️ Detected with {confidence:.2%} confidence")
        else:
            st.success(f"✅ Safe Content ({confidence:.2%} confidence)")

# 🎥 Video Upload Section
st.subheader("🎬 Upload a Video")
uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_video.read())
        temp_video_path = temp_file.name

    st.video(uploaded_video)
    st.write("🕵️ Processing video...")

    frames = extract_frames(temp_video_path, frame_interval=30)
    predictions = [float(model.predict(frame)[0][0]) for frame in frames]  # Convert to float
    violence_count = sum(1 for pred in predictions if pred > 0.5)
    non_violence_count = len(predictions) - violence_count

    # 📊 Show Results
    st.subheader("📊 Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🔴 Violence Frames", violence_count)
        st.error(f"{violence_count} frames detected as violence")

    with col2:
        st.metric("🟢 Non-Violence Frames", non_violence_count)
        st.success(f"{non_violence_count} frames detected as safe")

    # 📈 Plot a Bar Chart
    labels = ["Violence", "Non-Violence"]
    values = [violence_count, non_violence_count]
    fig, ax = plt.subplots()
    ax.bar(labels, values, color=["red", "green"])
    ax.set_ylabel("Number of Frames")
    ax.set_title("Violence Detection in Video")
    st.pyplot(fig)

    os.remove(temp_video_path)  # Cleanup temp file
