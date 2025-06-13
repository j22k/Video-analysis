import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'Emotion-detection'))

try:
    from EmotionDetection import detect_emotions_from_video
except ImportError as e:
    print(f"Error importing EmotionDetection module: {e}")
    sys.exit(1)
    
video_path = 'videoplayback.mp4'  # Replace with your video file path
# Call the function
results = detect_emotions_from_video(video_path)

# Print results
for frame_index, emotion, confidence in results:
    print(f"Frame {frame_index}: {emotion} ({confidence:.2%})")
