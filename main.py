import sys
import os
from collections import Counter
from moviepy import VideoFileClip

# Add the Emotion-detection directory to Python's path to allow imports
emotion_detection_path = os.path.join(os.path.dirname(__file__), 'Emotion-detection')
if emotion_detection_path not in sys.path:
    sys.path.insert(0, emotion_detection_path)

# --- Import Custom Modules ---
try:
    from EmotionDetection import detect_emotions_from_video
    from transcribe_audio import transcribe_audio
    from lanchain_deepseek import analyze_student_pitch
except ImportError as e:
    print(f"Fatal Error: Could not import the EmotionDetection module.")
    print(f"Details: {e}")
    sys.exit(1)

try:
    from Audio_analsys import analyze_audio
except ImportError as e:
    print(f"Fatal Error: Could not import the AudioAnalysis module.")
    print(f"Details: {e}")
    print("This might be due to a missing library like 'parselmouth' or 'librosa'.")
    sys.exit(1)

# --- Configuration ---
video_path = 'videoplayback.mp4'


def summarize_emotion_data(results, method="summary"):
    """
    Summarize emotion detection results using different methods
    
    Methods:
    - "summary": Overall statistics and dominant emotions
    - "intervals": Sample at regular intervals
    - "changes": Only show when emotion changes
    - "confidence_threshold": Only show high confidence detections
    """
    
    if not results:
        return "No emotion data available"
    
    if method == "summary":
        # Method 1: Statistical Summary
        emotions = [emotion for _, emotion, _ in results]
        confidences = [confidence for _, _, confidence in results]
        
        emotion_counts = Counter(emotions)
        avg_confidence = sum(confidences) / len(confidences)
        max_confidence = max(confidences)
        min_confidence = min(confidences)
        
        summary = f"""
--- Emotion Detection Summary ---
Total Frames Analyzed: {len(results)}
Average Confidence: {avg_confidence:.2%}
Confidence Range: {min_confidence:.2%} - {max_confidence:.2%}

Emotion Distribution:"""
        
        for emotion, count in emotion_counts.most_common():
            percentage = (count / len(results)) * 100
            summary += f"\n  {emotion}: {count} frames ({percentage:.1f}%)"
        
        # Show dominant emotion
        dominant_emotion = emotion_counts.most_common(1)[0]
        summary += f"\n\nDominant Emotion: {dominant_emotion[0]} ({dominant_emotion[1]} frames)"
        
        return summary
    
    elif method == "intervals":
        # Method 2: Sample at regular intervals (every 10th frame)
        interval = max(1, len(results) // 10)  # Show about 10 samples
        sampled_results = results[::interval]
        
        summary = f"--- Emotion Detection (Sampled Every {interval} Frames) ---\n"
        for frame_index, emotion, confidence in sampled_results:
            summary += f"Frame {frame_index}: {emotion} ({confidence:.2%})\n"
        
        return summary
    
    elif method == "changes":
        # Method 3: Only show when emotion changes
        if not results:
            return "No emotion changes detected"
        
        changes = [results[0]]  # Always include first frame
        current_emotion = results[0][1]
        
        for result in results[1:]:
            if result[1] != current_emotion:
                changes.append(result)
                current_emotion = result[1]
        
        summary = "--- Emotion Changes Only ---\n"
        for frame_index, emotion, confidence in changes:
            summary += f"Frame {frame_index}: Changed to {emotion} ({confidence:.2%})\n"
        
        return summary
    
    elif method == "confidence_threshold":
        # Method 4: Only show high confidence detections (>50%)
        threshold = 0.5
        high_confidence = [r for r in results if r[2] > threshold]
        
        if not high_confidence:
            summary = f"--- High Confidence Emotions (>{threshold:.0%}) ---\n"
            summary += "No high confidence detections found\n"
            # Show highest confidence detection
            best_result = max(results, key=lambda x: x[2])
            summary += f"Highest confidence: Frame {best_result[0]}: {best_result[1]} ({best_result[2]:.2%})"
        else:
            summary = f"--- High Confidence Emotions (>{threshold:.0%}) ---\n"
            for frame_index, emotion, confidence in high_confidence[:10]:  # Limit to 10
                summary += f"Frame {frame_index}: {emotion} ({confidence:.2%})\n"
            
            if len(high_confidence) > 10:
                summary += f"... and {len(high_confidence) - 10} more high confidence detections"
        
        return summary


def create_emotion_summary_for_analysis(results):
    """
    Create a concise emotion summary specifically for the AI analysis
    """
    if not results:
        return "No emotion data available"
    
    emotions = [emotion for _, emotion, _ in results]
    confidences = [confidence for _, _, confidence in results]
    
    emotion_counts = Counter(emotions)
    avg_confidence = sum(confidences) / len(confidences)
    
    # Create a concise summary for AI analysis
    summary = f"Emotion Analysis Summary:\n"
    summary += f"- Total frames: {len(results)}\n"
    summary += f"- Average confidence: {avg_confidence:.1%}\n"
    summary += f"- Dominant emotion: {emotion_counts.most_common(1)[0][0]}\n"
    summary += f"- Emotion distribution: "
    
    emotion_percentages = []
    for emotion, count in emotion_counts.most_common():
        percentage = (count / len(results)) * 100
        emotion_percentages.append(f"{emotion} ({percentage:.1f}%)")
    
    summary += ", ".join(emotion_percentages)
    
    return summary


# Extract audio from the video file
def extract_audio(video_path, output_folder="extracted_data"):
    """Extracts audio from a video file and saves it as a WAV file."""
    os.makedirs(output_folder, exist_ok=True)
    
    audio_path = os.path.join(output_folder, "audio.wav")
    try:
        print("Extracting audio from video...")
        video_clip = VideoFileClip(video_path)
        if video_clip.audio is None:
            print("Error: No audio track found in the video!")
            video_clip.close()
            return None
            
        # Write audio to a WAV file, which is ideal for speech_recognition
        video_clip.audio.write_audiofile(
            audio_path, 
            codec='pcm_s16le', # Standard codec for WAV files
            fps=16000 # Sample rate ideal for speech recognition
        )
        video_clip.close()
        
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            print(f"Successfully extracted audio to {audio_path}")
            return audio_path
        else:
            print("Error: Audio file was not created or is empty.")
            return None
            
    except Exception as e:
        print(f"An error occurred during audio extraction: {e}")
        return None


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
        
        # Print the results using different summarization methods
        if results:
            # Choose your preferred method here:
            # Options: "summary", "intervals", "changes", "confidence_threshold"
            
            print(summarize_emotion_data(results, method="summary"))
            
            # Uncomment below to see other methods:
            # print("\n" + summarize_emotion_data(results, method="intervals"))
            # print("\n" + summarize_emotion_data(results, method="changes"))
            # print("\n" + summarize_emotion_data(results, method="confidence_threshold"))
            
        else:
            print("\nNo emotions were detected in the video, or no faces were found.")
            
    except Exception as e:
        # Catch any other unexpected errors during processing
        print(f"\nAn error occurred during emotion detection: {e}")
        import traceback
        traceback.print_exc()
        
    try:
        # Attempt to extract audio if available
        audio_path = extract_audio(video_path)
        if audio_path:
            print(f"Audio extracted successfully to {audio_path}")
        else:
            print("No audio extracted due to an error.")
    except Exception as audio_error:
        print(f"An error occurred while extracting audio: {audio_error}")
        import traceback
        traceback.print_exc()
        
    try:
        audio_result = analyze_audio(audio_path)
        if audio_result:
            print("\n--- Audio Analysis Results ---")
            for key, value in audio_result.items():
                print(f"{key}: {value}")
        else:
            print("No audio analysis results available.")
    except Exception as audio_analysis_error:
        print(f"An error occurred during audio analysis: {audio_analysis_error}")
        import traceback
        traceback.print_exc()
    
    try:
        audio_text = transcribe_audio(audio_path)
        if audio_text:
            print("\n--- Audio Transcription Result ---")
            print(audio_text)
        else:
            print("No transcription result available.")
    except Exception as transcription_error:
        print(f"An error occurred during audio transcription: {transcription_error}")
        import traceback
        traceback.print_exc()
        
    try:
        # Use the summarized emotion data for analysis instead of raw results
        emotion_summary = create_emotion_summary_for_analysis(results)
        result = analyze_student_pitch(emotion_summary, audio_result, audio_text)
        if result:
            print("\n--- Student Pitch Performance Analysis ---")
            print(result)
        else:
            print("No analysis result available.")
    except Exception as analysis_error:
        print(f"An error occurred during student pitch performance analysis: {analysis_error}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_analysis()