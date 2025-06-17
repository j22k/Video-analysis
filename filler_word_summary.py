import whisper_timestamped as whisper
import json
import os

def analyze_filler_words(audio_path: str, model_size="tiny", device="cpu") -> dict:
    """
    Analyze filler words in a pre-extracted audio file using Whisper with timestamps.

    Args:
        audio_path (str): Path to the .wav audio file
        model_size (str): Size of Whisper model (e.g., 'tiny', 'base')
        device (str): 'cpu' or 'cuda'

    Returns:
        dict: {
            "clean_transcript": str,
            "filler_word_count": int,
            "filler_word_duration_secs": float,
            "total_audio_duration_secs": float,
            "filler_word_percentage": float
        }
    """
    audio = whisper.load_audio(audio_path)
    model = whisper.load_model(model_size, device=device)
    result = whisper.transcribe(model, audio, language="en", detect_disfluencies=True)

    total_duration = result["segments"][-1]["end"] if result["segments"] else 0.0

    filler_words = []
    filler_duration = 0.0
    for segment in result["segments"]:
        for word in segment["words"]:
            if "[*]" in word["text"]:
                start = word["start"]
                end = word["end"]
                filler_words.append({"start": start, "end": end})
                filler_duration += (end - start)

    cleaned_transcript = " ".join([
        word["text"] for segment in result["segments"]
        for word in segment["words"] if "[*]" not in word["text"]
    ])

    return {
        "clean_transcript": cleaned_transcript,
        "filler_word_count": len(filler_words),
        "filler_word_duration_secs": round(filler_duration, 2),
        "total_audio_duration_secs": round(total_duration, 2),
        "filler_word_percentage": round((filler_duration / total_duration) * 100, 2) if total_duration > 0 else 0.0
    }


# # ------------------ MAIN ------------------ #
# if __name__ == "__main__":
#     audio_path = "myaudio.wav"  # change this if needed

#     if not os.path.isfile(audio_path):
#         print(f"Audio file not found: {audio_path}")
#     else:
#         summary = analyze_filler_words(audio_path)
#         with open("abstract_transcript_summary_for_llm.json", "w", encoding="utf-8") as f:
#             json.dump(summary, f, indent=4)
#         print("Summary saved to abstract_transcript_summary_for_llm.json")
#         print(json.dumps(summary, indent=4))
