import os
import whisper_timestamped as whisper
ROOT_DIR = os.getcwd()

def transcribe_audio(file_path):
    """
    Transcribe audio using Whisper with timestamps.
    
    Args:
        file_path (str): Path to the audio file.
        
    Returns:
        list: List of tuples containing start time, end time, and text for each segment.
    """
    try:
        audio = whisper.load_audio(os.path.join(ROOT_DIR, file_path))
        model = whisper.load_model("tiny", device="cpu")
        result = whisper.transcribe(model, audio, language="en", detect_disfluencies=True)
        return result['text']
    
    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        return 'cant transcribe audio'
            

