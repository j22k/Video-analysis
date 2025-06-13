import cv2
import torch
import numpy as np
import mediapipe as mp
import os
from PIL import Image

# Import model modules (assuming same folder structure)
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from model.model import ResNet50, LSTMPyTorch
from model.utlis import pth_processing, get_box

def detect_emotions_from_video(video_path):
    """
    Detects emotions from a video file using ResNet50 + LSTM and returns a list of predicted emotions per clip.
    
    Parameters:
        video_path (str): Path to the input video file.

    Returns:
        emotions_list (list): List of tuples (frame_index, emotion_label, confidence).
    """

    # Load models
    model_dir = os.path.join(os.path.dirname(__file__), 'models')

    backbone_model = ResNet50(7, channels=3)
    backbone_model.load_state_dict(torch.load(os.path.join(model_dir, 'FER_static_ResNet50_AffectNet.pt')))
    backbone_model.eval()

    lstm_model = LSTMPyTorch()
    lstm_model.load_state_dict(torch.load(os.path.join(model_dir, 'FER_dinamic_LSTM_Aff-Wild2.pt')))
    lstm_model.eval()

    # Emotion dictionary
    DICT_EMO = {
        0: 'Neutral',
        1: 'Happiness',
        2: 'Sadness',
        3: 'Surprise',
        4: 'Fear',
        5: 'Disgust',
        6: 'Anger'
    }

    # MediaPipe face mesh setup
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Capture video
    cap = cv2.VideoCapture(video_path)
    emotions_list = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            h, w = frame.shape[:2]
            fl = results.multi_face_landmarks[0]
            startX, startY, endX, endY = get_box(fl, w, h)

            # Clip and check bounds
            startX, startY = max(0, int(startX)), max(0, int(startY))
            endX, endY = min(w, int(endX)), min(h, int(endY))

            face = frame_rgb[startY:endY, startX:endX]
            if face.size > 0:
                face_pil = Image.fromarray(face)
                face_processed = pth_processing(face_pil)

                with torch.no_grad():
                    features = torch.nn.functional.relu(
                        backbone_model.extract_features(face_processed)
                    ).detach().cpu().numpy()

                    lstm_features = [features] * 10
                    lstm_f = torch.from_numpy(np.vstack(lstm_features))
                    lstm_f = torch.unsqueeze(lstm_f, 0)

                    output = lstm_model(lstm_f).detach().cpu().numpy()
                    emotion_idx = np.argmax(output)
                    emotion = DICT_EMO.get(emotion_idx, "Unknown")
                    confidence = float(output[0][emotion_idx])

                    emotions_list.append((frame_idx, emotion, confidence))

        frame_idx += 1

    cap.release()
    return emotions_list
