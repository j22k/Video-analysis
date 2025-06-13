# File: Emotion-detection/EmotionDetection.py

import cv2
import torch
import numpy as np
import mediapipe as mp
import os
from PIL import Image

# Import model definitions from the model.py file
from model.model import ResNet50, LSTMPyTorch
from model.utlis import pth_processing, get_box

def detect_emotions_from_video(video_path):
    """
    Detects emotions from a video file using ResNet50 + LSTM and returns a list of results.
    
    Parameters:
        video_path (str): Path to the input video file.

    Returns:
        emotions_list (list): List of tuples (frame_index, emotion_label, confidence).
    """
    print("Loading models...")
    # Get the directory where the model files are located
    model_dir = os.path.join(os.path.dirname(__file__), 'model')

    # Load the ResNet50 model for feature extraction
    backbone_model = ResNet50(7, channels=3)
    backbone_model.load_state_dict(torch.load(os.path.join(model_dir, 'FER_static_ResNet50_AffectNet.pt'), weights_only=True))
    backbone_model.eval()

    # Load the LSTM model for emotion classification
    lstm_model = LSTMPyTorch()
    lstm_model.load_state_dict(torch.load(os.path.join(model_dir, 'FER_dinamic_LSTM_Aff-Wild2.pt'), weights_only=True))
    lstm_model.eval()
    print("Models loaded successfully.")

    # Emotion dictionary to map model output to labels
    DICT_EMO = {
        0: 'Neutral', 1: 'Happiness', 2: 'Sadness', 3: 'Surprise',
        4: 'Fear', 5: 'Disgust', 6: 'Anger'
    }

    # Initialize MediaPipe Face Mesh for face detection
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return []

    emotions_list = []
    frame_idx = 0

    print("Processing video frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video

        # Convert frame from BGR to RGB (required for MediaPipe)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            h, w = frame.shape[:2]
            # Get the bounding box of the detected face
            fl = results.multi_face_landmarks[0]
            startX, startY, endX, endY = get_box(fl, w, h)

            # Ensure coordinates are within frame boundaries
            startX, startY = max(0, int(startX)), max(0, int(startY))
            endX, endY = min(w, int(endX)), min(h, int(endY))

            face = frame_rgb[startY:endY, startX:endX]
            
            # Process only if a valid face crop is obtained
            if face.size > 0:
                face_pil = Image.fromarray(face)
                face_processed = pth_processing(face_pil)

                with torch.no_grad():
                    # 1. Extract features using the ResNet backbone
                    features = torch.nn.functional.relu(
                        backbone_model.extract_features(face_processed)
                    ).detach().cpu().numpy()

                    # 2. Prepare features for the LSTM (it expects a sequence)
                    lstm_features = [features] * 10
                    lstm_f = torch.from_numpy(np.vstack(lstm_features))
                    lstm_f = torch.unsqueeze(lstm_f, 0)

                    # 3. Get emotion prediction from the LSTM model
                    output = lstm_model(lstm_f).detach().cpu().numpy()
                    emotion_idx = np.argmax(output)
                    emotion = DICT_EMO.get(emotion_idx, "Unknown")
                    confidence = float(output[0][emotion_idx])

                    emotions_list.append((frame_idx, emotion, confidence))

        frame_idx += 1

    cap.release()
    print(f"Video processing finished. Detected emotions in {len(emotions_list)} frames.")
    return emotions_list