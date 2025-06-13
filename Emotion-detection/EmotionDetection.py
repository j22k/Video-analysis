import tkinter as tk
from tkinter import ttk
import cv2
import PIL.Image, PIL.ImageTk
import torch
import numpy as np
from PIL import Image
import mediapipe as mp
import os

# Import our emotion detection models
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from model.model import ResNet50, LSTMPyTorch
from model.utlis import pth_processing, get_box

class EmotionDetectionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Emotion Detection")
        
        # Initialize models
        self.init_models()
        
        # Initialize MediaPipe face mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Emotion dictionary
        self.DICT_EMO = {
            0: 'Neutral', 
            1: 'Happiness', 
            2: 'Sadness', 
            3: 'Surprise', 
            4: 'Fear', 
            5: 'Disgust', 
            6: 'Anger'
        }
        
        # Create GUI elements
        self.create_widgets()
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        self.update()

    def init_models(self):
        # Load the models
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        
        self.backbone_model = ResNet50(7, channels=3)
        self.backbone_model.load_state_dict(torch.load(os.path.join(model_dir, 'FER_static_ResNet50_AffectNet.pt')))
        self.backbone_model.eval()

        self.lstm_model = LSTMPyTorch()
        self.lstm_model.load_state_dict(torch.load(os.path.join(model_dir, 'FER_dinamic_LSTM_Aff-Wild2.pt')))
        self.lstm_model.eval()

    def create_widgets(self):
        # Create main container
        self.container = ttk.Frame(self.window, padding="10")
        self.container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create video label
        self.video_label = ttk.Label(self.container)
        self.video_label.grid(row=0, column=0, columnspan=2, pady=5)

        # Create emotion label
        style = ttk.Style()
        style.configure('Emotion.TLabel', font=('Helvetica', 14, 'bold'))
        self.emotion_label = ttk.Label(self.container, text="Detecting...", style='Emotion.TLabel')
        self.emotion_label.grid(row=1, column=0, columnspan=2, pady=10)

        # Create confidence progressbar
        self.confidence_var = tk.DoubleVar()
        self.confidence_bar = ttk.Progressbar(
            self.container,
            orient="horizontal",
            length=300,
            mode="determinate",
            variable=self.confidence_var
        )
        self.confidence_bar.grid(row=2, column=0, columnspan=2, pady=5)
        
        # Create quit button
        self.quit_button = ttk.Button(self.container, text="Quit", command=self.quit)
        self.quit_button.grid(row=3, column=0, columnspan=2, pady=10)

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                h, w = frame.shape[:2]
                fl = results.multi_face_landmarks[0]
                
                # Get face box
                startX, startY, endX, endY = get_box(fl, w, h)
                
                # Draw rectangle around face
                cv2.rectangle(frame_rgb, (int(startX), int(startY)), 
                            (int(endX), int(endY)), (0, 255, 0), 2)
                
                # Extract face and process
                face = frame_rgb[int(startY):int(endY), int(startX):int(endX)]
                if face.size > 0:
                    # Convert to PIL Image
                    face_pil = Image.fromarray(face)
                    face_processed = pth_processing(face_pil)
                    
                    # Emotion detection
                    with torch.no_grad():
                        features = torch.nn.functional.relu(
                            self.backbone_model.extract_features(face_processed)
                        ).detach().cpu().numpy()
                        
                        # Prepare LSTM input
                        lstm_features = [features] * 10
                        lstm_f = torch.from_numpy(np.vstack(lstm_features))
                        lstm_f = torch.unsqueeze(lstm_f, 0)
                        
                        output = self.lstm_model(lstm_f).detach().cpu().numpy()
                        
                        # Get prediction
                        emotion_idx = np.argmax(output)
                        emotion = self.DICT_EMO.get(emotion_idx, "Unknown")
                        confidence = float(output[0][emotion_idx])
                        
                        # Update GUI
                        self.emotion_label.configure(text=f"Emotion: {emotion}")
                        self.confidence_var.set(confidence * 100)
            
            # Convert frame to PhotoImage
            img = PIL.Image.fromarray(frame_rgb)
            img = img.resize((640, 480))
            imgtk = PIL.ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        
        # Schedule next update
        self.window.after(10, self.update)

    def quit(self):
        self.cap.release()
        self.window.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionDetectionApp(root)
    root.mainloop()
