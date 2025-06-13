# import librosa
# import numpy as np
# from parselmouth import Sound
# import parselmouth.praat as praat

# def analyze_audio_quality(audio_path):
#     # Load audio
#     y, sr = librosa.load(audio_path)
    
#     # Extract features
#     features = {
#         'pitch_variation': calculate_pitch_variation(audio_path),
#         'speaking_rate': calculate_speaking_rate(y, sr),
#         'volume_consistency': calculate_volume_consistency(y),
#         'pause_analysis': detect_pauses(y, sr),
#         'voice_quality': assess_voice_quality(audio_path)
#     }
    
#     return features

# def calculate_pitch_variation(audio_path):
#     sound = Sound(audio_path)
#     pitch = sound.to_pitch()
#     pitch_values = pitch.selected_array['frequency']
#     pitch_values = pitch_values[pitch_values != 0]  # Remove unvoiced frames
    
#     return {
#         'mean_pitch': np.mean(pitch_values),
#         'pitch_std': np.std(pitch_values),
#         'pitch_range': np.max(pitch_values) - np.min(pitch_values)
#     }


import librosa
import numpy as np
from parselmouth import Sound
import parselmouth.praat as praat

def analyze_audio_quality(audio_path):
    # Load audio
    y, sr = librosa.load(audio_path)

    # Extract features
    features = {
        'pitch_variation': calculate_pitch_variation(audio_path),
        'speaking_rate': calculate_speaking_rate(y, sr),
        'volume_consistency': calculate_volume_consistency(y),
        'pause_analysis': detect_pauses(y, sr),
        'voice_quality': assess_voice_quality(audio_path)
    }

    return features

def calculate_pitch_variation(audio_path):
    sound = Sound(audio_path)
    pitch = sound.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    pitch_values = pitch_values[pitch_values != 0]  # Remove unvoiced frames

    return {
        'mean_pitch': float(np.mean(pitch_values)),
        'pitch_std': float(np.std(pitch_values)),
        'pitch_range': float(np.max(pitch_values) - np.min(pitch_values))
    }

# Placeholder functions
def calculate_speaking_rate(y, sr):
    duration = librosa.get_duration(y=y, sr=sr)
    non_silent_intervals = librosa.effects.split(y, top_db=25)
    total_voice_duration = sum((end - start) for start, end in non_silent_intervals) / sr
    speaking_rate = len(non_silent_intervals) / total_voice_duration if total_voice_duration > 0 else 0
    return round(speaking_rate, 2)

def calculate_volume_consistency(y):
    rms = librosa.feature.rms(y=y)[0]
    return {
        'rms_mean': float(np.mean(rms)),
        'rms_std': float(np.std(rms))
    }

def detect_pauses(y, sr):
    intervals = librosa.effects.split(y, top_db=25)
    pause_durations = []
    for i in range(1, len(intervals)):
        prev_end = intervals[i-1][1]
        curr_start = intervals[i][0]
        pause = (curr_start - prev_end) / sr
        if pause > 0.2:  # Consider pauses >200ms
            pause_durations.append(pause)
    return {
        'num_pauses': len(pause_durations),
        'avg_pause_duration': float(np.mean(pause_durations)) if pause_durations else 0
    }

def assess_voice_quality(audio_path):
    sound = Sound(audio_path)
    point_process = praat.call(sound, "To PointProcess (periodic, cc)", 75, 500)
    jitter = praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer = praat.call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    return {
        'jitter': round(jitter, 5),
        'shimmer': round(shimmer, 5)
    }

# ✅ Run analysis and print output
if __name__ == "__main__":
    audio_path = r"C:\Zoftcares\Video Extraction\extracted_audio.wav"
    print("✅ Script started")
    try:
        features = analyze_audio_quality(audio_path)
        print("🎯 Audio Quality Analysis:")
        for key, value in features.items():
            print(f"\n🔹 {key.capitalize()}:")
            if isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    print(f"   - {sub_key}: {sub_val}")
            else:
                print(f"   {value}")
    except Exception as e:
        print("❌ Error:", e)
