import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
from PIL import Image

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(p1, p2):
    """Calculates the angle of the line connecting two points relative to the horizontal."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    angle_rad = math.atan2(dy, dx)
    return math.degrees(angle_rad)

# --- Streamlit UI Setup ---
st.title("🧍‍♀️ Posture & Symmetry Analyzer")
st.write("Upload an image of yourself facing the camera to get a symmetry score!")

# 1. Create the File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 2. Convert the uploaded web image into an OpenCV format
    # Streamlit uses PIL (Python Imaging Library), OpenCV uses Numpy arrays in BGR format
    pil_image = Image.open(uploaded_file).convert('RGB')
    image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    h, w, _ = image_bgr.shape
    image_rgb_mp = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # MediaPipe needs RGB

    # 3. Run MediaPipe Processing
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(image_rgb_mp)

        if not results.pose_landmarks:
            st.error("No person detected in the image. Please try another one!")
        else:
            # Draw the skeletal overlay
            mp_drawing.draw_landmarks(
                image_bgr, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

            # Extract Landmarks
            landmarks = results.pose_landmarks.landmark
            l_shoulder = (int(landmarks[11].x * w), int(landmarks[11].y * h))
            r_shoulder = (int(landmarks[12].x * w), int(landmarks[12].y * h))
            l_hip = (int(landmarks[23].x * w), int(landmarks[23].y * h))
            r_hip = (int(landmarks[24].x * w), int(landmarks[24].y * h))

            # Calculate Angles & Score
            shoulder_angle = calculate_angle(l_shoulder, r_shoulder)
            hip_angle = calculate_angle(l_hip, r_hip)

            max_deviation = 45 
            shoulder_dev = abs(shoulder_angle) if abs(shoulder_angle) < 90 else abs(180 - abs(shoulder_angle))
            hip_dev = abs(hip_angle) if abs(hip_angle) < 90 else abs(180 - abs(hip_angle))
            
            avg_deviation = (shoulder_dev + hip_dev) / 2
            symmetry_score = max(0, 100 - (avg_deviation / max_deviation * 100))

            # Output Text to Image (using the smaller font scale we figured out earlier)
            font_scale = max(0.4, w / 1500) # Scales font slightly based on image width
            thickness = max(1, int(w / 800))
            
            cv2.putText(image_bgr, f"Shoulder: {shoulder_dev:.1f} deg", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
            cv2.putText(image_bgr, f"Hip: {hip_dev:.1f} deg", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
            
            color = (0, 255, 0) if symmetry_score > 85 else (0, 0, 255)
            cv2.putText(image_bgr, f"Score: {symmetry_score:.1f}%", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.2, color, thickness + 1)

            cv2.line(image_bgr, l_shoulder, r_shoulder, (255, 255, 0), 2)
            cv2.line(image_bgr, l_hip, r_hip, (255, 255, 0), 2)

            # 4. Display the final image in Streamlit
            # Convert back to RGB for Streamlit to display colors correctly
            final_image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            st.success(f"Analysis Complete! Your Symmetry Score is {symmetry_score:.1f}%")
            # use_container_width makes sure the image fits nicely on the web page
            st.image(final_image_rgb, caption="Analyzed Posture", use_container_width=True)