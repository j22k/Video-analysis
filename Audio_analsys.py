# File: Audio-analysis/AudioAnalysis.py

import librosa
import numpy as np
from parselmouth import Sound
import parselmouth.praat as praat
import json
import os
from typing import Dict, Any

class AudioQualityAnalyzer:
    """
    Comprehensive audio quality analyzer that extracts multiple features
    from audio files and returns structured results.
    """
    
    def __init__(self, audio_path: str):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found at: {audio_path}")
        self.audio_path = audio_path
        self.y = None
        self.sr = None
        self.sound = None
        self._load_audio()
    
    def _load_audio(self):
        try:
            self.y, self.sr = librosa.load(self.audio_path, sr=None)
            self.sound = Sound(self.audio_path)
            print(f"✅ Audio loaded successfully: {os.path.basename(self.audio_path)}")
        except Exception as e:
            raise IOError(f"Failed to load audio file: {str(e)}")
    
    def calculate_pitch_variation(self) -> Dict[str, float]:
        try:
            pitch = self.sound.to_pitch()
            pitch_values = pitch.selected_array['frequency']
            pitch_values = pitch_values[pitch_values != 0]
            
            if len(pitch_values) == 0: return self._default_pitch_metrics()
            
            total_frames = len(pitch.selected_array['frequency'])
            voiced_percentage = (len(pitch_values) / total_frames) * 100 if total_frames > 0 else 0.0
            
            return {
                'mean_pitch': float(np.mean(pitch_values)), 'pitch_std': float(np.std(pitch_values)),
                'pitch_range': float(np.max(pitch_values) - np.min(pitch_values)),
                'pitch_median': float(np.median(pitch_values)),
                'pitch_iqr': float(np.percentile(pitch_values, 75) - np.percentile(pitch_values, 25)),
                'voiced_frames_percentage': float(voiced_percentage)
            }
        except Exception as e:
            print(f"⚠️ Error in pitch calculation: {str(e)}")
            return self._default_pitch_metrics()

    def _default_pitch_metrics(self) -> Dict[str, float]:
        return {'mean_pitch': 0.0, 'pitch_std': 0.0, 'pitch_range': 0.0, 'pitch_median': 0.0, 'pitch_iqr': 0.0, 'voiced_frames_percentage': 0.0}

    def calculate_speaking_rate(self) -> Dict[str, float]:
        try:
            duration = librosa.get_duration(y=self.y, sr=self.sr)
            non_silent_intervals = librosa.effects.split(self.y, top_db=25)
            
            if len(non_silent_intervals) == 0:
                return {'total_duration': float(duration), 'speech_duration': 0.0, 'silence_duration': float(duration), 'speech_rate': 0.0, 'speech_to_silence_ratio': 0.0, 'num_speech_segments': 0}
            
            speech_duration = sum((end - start) for start, end in non_silent_intervals) / self.sr
            silence_duration = duration - speech_duration
            speaking_rate = len(non_silent_intervals) / speech_duration if speech_duration > 0 else 0
            speech_to_silence_ratio = speech_duration / silence_duration if silence_duration > 0 else float('inf')
            
            return {
                'total_duration': float(duration), 'speech_duration': float(speech_duration),
                'silence_duration': float(silence_duration), 'speech_rate': float(speaking_rate),
                'speech_to_silence_ratio': float(speech_to_silence_ratio), 'num_speech_segments': len(non_silent_intervals)
            }
        except Exception as e:
            print(f"⚠️ Error in speaking rate calculation: {str(e)}")
            return {'total_duration': 0.0, 'speech_duration': 0.0, 'silence_duration': 0.0, 'speech_rate': 0.0, 'speech_to_silence_ratio': 0.0, 'num_speech_segments': 0}

    def calculate_volume_consistency(self) -> Dict[str, float]:
        try:
            rms = librosa.feature.rms(y=self.y)[0]
            db_values = librosa.amplitude_to_db(rms, ref=np.max)
            dynamic_range = np.max(db_values) - np.min(db_values)
            return {'rms_mean': float(np.mean(rms)), 'rms_std': float(np.std(rms)), 'dynamic_range_db': float(dynamic_range)}
        except Exception as e:
            print(f"⚠️ Error in volume consistency calculation: {str(e)}")
            return {'rms_mean': 0.0, 'rms_std': 0.0, 'dynamic_range_db': 0.0}
    
    def detect_pauses(self) -> Dict[str, float]:
        try:
            intervals = librosa.effects.split(self.y, top_db=25)
            if len(intervals) <= 1: return self._default_pause_metrics()
            
            pause_durations = [(intervals[i][0] - intervals[i-1][1]) / self.sr for i in range(1, len(intervals)) if (intervals[i][0] - intervals[i-1][1]) / self.sr > 0.1]
            if not pause_durations: return self._default_pause_metrics()
            
            return {
                'num_pauses': len(pause_durations), 'total_pause_duration': float(sum(pause_durations)),
                'avg_pause_duration': float(np.mean(pause_durations)), 'max_pause_duration': float(max(pause_durations)),
            }
        except Exception as e:
            print(f"⚠️ Error in pause detection: {str(e)}")
            return self._default_pause_metrics()

    def _default_pause_metrics(self) -> Dict[str, float]:
        return {'num_pauses': 0, 'total_pause_duration': 0.0, 'avg_pause_duration': 0.0, 'max_pause_duration': 0.0}

    def assess_voice_quality(self) -> Dict[str, float]:
        try:
            point_process = praat.call(self.sound, "To PointProcess (periodic, cc)", 75, 500)
            jitter_local = praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer_local = praat.call([self.sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            harmonicity = praat.call(self.sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr = praat.call(harmonicity, "Get mean", 0, 0)
            return {'jitter_local': float(jitter_local), 'shimmer_local': float(shimmer_local), 'harmonics_to_noise_ratio': float(hnr)}
        except Exception as e:
            print(f"⚠️ Error in voice quality assessment: {str(e)}")
            return {'jitter_local': 0.0, 'shimmer_local': 0.0, 'harmonics_to_noise_ratio': 0.0}

    def analyze_all(self) -> Dict[str, Any]:
        print("🎯 Starting comprehensive audio analysis...")
        results = {
            'file_info': {'filename': os.path.basename(self.audio_path), 'duration_seconds': float(librosa.get_duration(y=self.y, sr=self.sr))},
            'pitch_analysis': self.calculate_pitch_variation(),
            'speech_timing': self.calculate_speaking_rate(),
            'volume_analysis': self.calculate_volume_consistency(),
            'pause_analysis': self.detect_pauses(),
            'voice_quality': self.assess_voice_quality(),
        }
        print("✅ Audio analysis completed successfully!")
        return results

# ==============================================================================
#  Main function to be called from other scripts
# ==============================================================================
def analyze_audio(audio_path: str) -> Dict[str, Any]:
    """
    High-level function to instantiate the analyzer, run all analyses,
    and return the results.

    Args:
        audio_path (str): The path to the audio file to be analyzed.

    Returns:
        A dictionary containing the full analysis results, or an error dictionary.
    """
    try:
        analyzer = AudioQualityAnalyzer(audio_path)
        results = analyzer.analyze_all()
        return results
    except Exception as e:
        print(f"❌ Critical error during audio analysis: {str(e)}")
        return {"error": f"Audio analysis failed: {str(e)}"}