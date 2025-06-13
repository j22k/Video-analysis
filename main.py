# File: main.py

import sys
import os

# Add the Emotion-detection directory to Python's path to allow imports
emotion_detection_path = os.path.join(os.path.dirname(__file__), 'Emotion-detection')
if emotion_detection_path not in sys.path:
    sys.path.insert(0, emotion_detection_path)

try:
    # Import the main function from your module
    from EmotionDetection import detect_emotions_from_video
except ImportError as e:
    print(f"Fatal Error: Could not import the EmotionDetection module.")
    print(f"Details: {e}")
    print(f"Please ensure '__init__.py' exists in the 'Emotion-detection' folder.")
    print(f"Attempted to add path: {emotion_detection_path}")
    sys.exit(1)

# --- Configuration ---
video_path = 'videoplayback.mp4'  # IMPORTANT: Place your video file in the same directory as this script

def run_analysis():
    # Check if the video file exists before starting
    if not os.path.exists(video_path):
        print(f"Fatal Error: Video file not found at '{os.path.abspath(video_path)}'")
        print("Please make sure the video file is in the correct location.")
        sys.exit(1)

    print("Starting emotion detection process...")
    try:
        # Call the function to get the results
        results = detect_emotions_from_video(video_path)
        
        # Print the results
        if results:
            print(f"\n--- Emotion Detection Results ---")
            for frame_index, emotion, confidence in results:
                print(f"Frame {frame_index}: Emotion = {emotion} (Confidence: {confidence:.2%})")
        else:
            print("\nNo emotions were detected in the video, or no faces were found.")
            
    except Exception as e:
        # Catch any other unexpected errors during processing
        print(f"\nAn error occurred during emotion detection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_analysis()