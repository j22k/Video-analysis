import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import sys
import os
import threading
from collections import Counter
from moviepy import VideoFileClip
import io
import contextlib

# Add the Emotion-detection directory to Python's path to allow imports
emotion_detection_path = os.path.join(os.path.dirname(__file__), 'Emotion-detection')
if emotion_detection_path not in sys.path:
    sys.path.insert(0, emotion_detection_path)

# --- Import Custom Modules ---
try:
    from EmotionDetection import detect_emotions_from_video
    from transcribe_audio import transcribe_audio
    from lanchain_deepseek import analyze_student_pitch
    from Audio_analsys import analyze_audio
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")


class StudentPitchAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Student Pitch Performance Analyzer")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Variables
        self.video_path = tk.StringVar()
        self.analysis_results = {}
        self.is_analyzing = False
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Student Pitch Performance Analyzer", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # File selection section
        file_frame = ttk.LabelFrame(main_frame, text="Video File Selection", padding="10")
        file_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        ttk.Label(file_frame, text="Video File:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        
        self.file_entry = ttk.Entry(file_frame, textvariable=self.video_path, width=50)
        self.file_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        
        ttk.Button(file_frame, text="Browse", command=self.browse_file).grid(row=0, column=2)
        
        # Analyze button
        self.analyze_button = ttk.Button(file_frame, text="Start Analysis", 
                                        command=self.start_analysis, style='Accent.TButton')
        self.analyze_button.grid(row=1, column=0, columnspan=3, pady=(10, 0))
        
        # Progress bar
        self.progress = ttk.Progressbar(file_frame, mode='indeterminate')
        self.progress.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Main content area
        content_frame = ttk.Frame(main_frame)
        content_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        content_frame.columnconfigure(0, weight=2)
        content_frame.columnconfigure(1, weight=1)
        content_frame.rowconfigure(0, weight=1)
        
        # Results section (left side - larger)
        results_frame = ttk.LabelFrame(content_frame, text="Analysis Results", padding="10")
        results_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Notebook for different result tabs
        self.results_notebook = ttk.Notebook(results_frame)
        self.results_notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Emotion Analysis Tab
        self.emotion_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.emotion_frame, text="Emotion Analysis")
        
        self.emotion_text = scrolledtext.ScrolledText(self.emotion_frame, wrap=tk.WORD, 
                                                     height=15, font=('Consolas', 10))
        self.emotion_text.pack(fill=tk.BOTH, expand=True)
        
        # Audio Analysis Tab
        self.audio_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.audio_frame, text="Audio Analysis")
        
        self.audio_text = scrolledtext.ScrolledText(self.audio_frame, wrap=tk.WORD, 
                                                   height=15, font=('Consolas', 10))
        self.audio_text.pack(fill=tk.BOTH, expand=True)
        
        # Transcription Tab
        self.transcription_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.transcription_frame, text="Transcription")
        
        self.transcription_text = scrolledtext.ScrolledText(self.transcription_frame, wrap=tk.WORD, 
                                                           height=15, font=('Arial', 10))
        self.transcription_text.pack(fill=tk.BOTH, expand=True)
        
        # Final Analysis Tab
        self.final_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.final_frame, text="Performance Analysis")
        
        self.final_text = scrolledtext.ScrolledText(self.final_frame, wrap=tk.WORD, 
                                                   height=15, font=('Arial', 10))
        self.final_text.pack(fill=tk.BOTH, expand=True)
        
        # Console output section (right side - smaller)
        console_frame = ttk.LabelFrame(content_frame, text="Console Output", padding="10")
        console_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        console_frame.columnconfigure(0, weight=1)
        console_frame.rowconfigure(0, weight=1)
        
        self.console_text = scrolledtext.ScrolledText(console_frame, wrap=tk.WORD, 
                                                     height=20, width=40, 
                                                     font=('Consolas', 9), 
                                                     bg='#2d2d2d', fg='#ffffff')
        self.console_text.pack(fill=tk.BOTH, expand=True)
        
        # Clear buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=3, column=0, columnspan=3, pady=(10, 0))
        
        ttk.Button(buttons_frame, text="Clear Results", 
                  command=self.clear_results).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(buttons_frame, text="Clear Console", 
                  command=self.clear_console).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(buttons_frame, text="Save Results", 
                  command=self.save_results).pack(side=tk.LEFT)
        
    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.video_path.set(filename)
    
    def log_to_console(self, message):
        """Add message to console output"""
        self.console_text.insert(tk.END, message + "\n")
        self.console_text.see(tk.END)
        self.root.update_idletasks()
    
    def clear_console(self):
        self.console_text.delete(1.0, tk.END)
    
    def clear_results(self):
        self.emotion_text.delete(1.0, tk.END)
        self.audio_text.delete(1.0, tk.END)
        self.transcription_text.delete(1.0, tk.END)
        self.final_text.delete(1.0, tk.END)
        self.analysis_results = {}
    
    def save_results(self):
        if not self.analysis_results:
            messagebox.showwarning("No Results", "No analysis results to save.")
            return
            
        filename = filedialog.asksaveasfilename(
            title="Save Analysis Results",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("Student Pitch Performance Analysis Results\n")
                    f.write("=" * 50 + "\n\n")
                    
                    for section, content in self.analysis_results.items():
                        f.write(f"{section}\n")
                        f.write("-" * len(section) + "\n")
                        f.write(str(content) + "\n\n")
                
                messagebox.showinfo("Success", f"Results saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {str(e)}")
    
    def start_analysis(self):
        if not self.video_path.get():
            messagebox.showerror("Error", "Please select a video file first.")
            return
        
        if not os.path.exists(self.video_path.get()):
            messagebox.showerror("Error", "Selected video file does not exist.")
            return
        
        if self.is_analyzing:
            messagebox.showwarning("Warning", "Analysis is already in progress.")
            return
        
        # Clear previous results
        self.clear_results()
        self.clear_console()
        
        # Start analysis in a separate thread
        self.is_analyzing = True
        self.analyze_button.config(state='disabled')
        self.progress.start(10)
        
        thread = threading.Thread(target=self.run_analysis)
        thread.daemon = True
        thread.start()
    
    def run_analysis(self):
        try:
            video_path = self.video_path.get()
            
            self.log_to_console("Starting emotion detection process...")
            
            # Emotion Detection
            try:
                self.log_to_console("Analyzing emotions in video...")
                results = detect_emotions_from_video(video_path)
                
                if results:
                    emotion_summary = self.summarize_emotion_data(results, method="summary")
                    self.analysis_results["Emotion Analysis"] = emotion_summary
                    
                    # Update UI in main thread
                    self.root.after(0, lambda: self.emotion_text.insert(tk.END, emotion_summary))
                    self.log_to_console("Emotion detection completed successfully.")
                else:
                    error_msg = "No emotions were detected in the video, or no faces were found."
                    self.analysis_results["Emotion Analysis"] = error_msg
                    self.root.after(0, lambda: self.emotion_text.insert(tk.END, error_msg))
                    self.log_to_console(error_msg)
                    
            except Exception as e:
                error_msg = f"Error during emotion detection: {str(e)}"
                self.log_to_console(error_msg)
                self.root.after(0, lambda: self.emotion_text.insert(tk.END, error_msg))
            
            # Audio Extraction
            try:
                self.log_to_console("Extracting audio from video...")
                audio_path = self.extract_audio(video_path)
                if audio_path:
                    self.log_to_console(f"Audio extracted successfully to {audio_path}")
                else:
                    self.log_to_console("No audio extracted due to an error.")
                    return
            except Exception as e:
                self.log_to_console(f"Error during audio extraction: {str(e)}")
                return
            
            # Audio Analysis
            try:
                self.log_to_console("Analyzing audio characteristics...")
                audio_result = analyze_audio(audio_path)
                if audio_result:
                    audio_text = "Audio Analysis Results:\n" + "\n".join([f"{k}: {v}" for k, v in audio_result.items()])
                    self.analysis_results["Audio Analysis"] = audio_text
                    self.root.after(0, lambda: self.audio_text.insert(tk.END, audio_text))
                    self.log_to_console("Audio analysis completed successfully.")
                else:
                    error_msg = "No audio analysis results available."
                    self.analysis_results["Audio Analysis"] = error_msg
                    self.root.after(0, lambda: self.audio_text.insert(tk.END, error_msg))
                    self.log_to_console(error_msg)
            except Exception as e:
                error_msg = f"Error during audio analysis: {str(e)}"
                self.log_to_console(error_msg)
                self.root.after(0, lambda: self.audio_text.insert(tk.END, error_msg))
                audio_result = None
            
            # Audio Transcription
            try:
                self.log_to_console("Transcribing audio...")
                audio_text = transcribe_audio(audio_path)
                if audio_text:
                    self.analysis_results["Transcription"] = audio_text
                    self.root.after(0, lambda: self.transcription_text.insert(tk.END, audio_text))
                    self.log_to_console("Audio transcription completed successfully.")
                else:
                    error_msg = "No transcription result available."
                    self.analysis_results["Transcription"] = error_msg
                    self.root.after(0, lambda: self.transcription_text.insert(tk.END, error_msg))
                    self.log_to_console(error_msg)
            except Exception as e:
                error_msg = f"Error during audio transcription: {str(e)}"
                self.log_to_console(error_msg)
                self.root.after(0, lambda: self.transcription_text.insert(tk.END, error_msg))
                audio_text = None
            
            # Final Analysis
            try:
                self.log_to_console("Performing final pitch performance analysis...")
                if 'results' in locals():
                    emotion_summary = self.create_emotion_summary_for_analysis(results)
                else:
                    emotion_summary = "No emotion data available"
                
                final_result = analyze_student_pitch(emotion_summary, audio_result, audio_text)
                if final_result:
                    self.analysis_results["Performance Analysis"] = final_result
                    self.root.after(0, lambda: self.final_text.insert(tk.END, final_result))
                    self.log_to_console("Student pitch performance analysis completed successfully.")
                else:
                    error_msg = "No final analysis result available."
                    self.analysis_results["Performance Analysis"] = error_msg
                    self.root.after(0, lambda: self.final_text.insert(tk.END, error_msg))
                    self.log_to_console(error_msg)
            except Exception as e:
                error_msg = f"Error during final analysis: {str(e)}"
                self.log_to_console(error_msg)
                self.root.after(0, lambda: self.final_text.insert(tk.END, error_msg))
            
            self.log_to_console("Analysis completed!")
            
        except Exception as e:
            error_msg = f"Fatal error during analysis: {str(e)}"
            self.log_to_console(error_msg)
            messagebox.showerror("Error", error_msg)
        
        finally:
            # Re-enable UI elements
            self.root.after(0, self.analysis_complete)
    
    def analysis_complete(self):
        self.is_analyzing = False
        self.analyze_button.config(state='normal')
        self.progress.stop()
    
    def summarize_emotion_data(self, results, method="summary"):
        """Summarize emotion detection results"""
        if not results:
            return "No emotion data available"
        
        if method == "summary":
            emotions = [emotion for _, emotion, _ in results]
            confidences = [confidence for _, _, confidence in results]
            
            emotion_counts = Counter(emotions)
            avg_confidence = sum(confidences) / len(confidences)
            max_confidence = max(confidences)
            min_confidence = min(confidences)
            
            summary = f"""--- Emotion Detection Summary ---
Total Frames Analyzed: {len(results)}
Average Confidence: {avg_confidence:.2%}
Confidence Range: {min_confidence:.2%} - {max_confidence:.2%}

Emotion Distribution:"""
            
            for emotion, count in emotion_counts.most_common():
                percentage = (count / len(results)) * 100
                summary += f"\n  {emotion}: {count} frames ({percentage:.1f}%)"
            
            dominant_emotion = emotion_counts.most_common(1)[0]
            summary += f"\n\nDominant Emotion: {dominant_emotion[0]} ({dominant_emotion[1]} frames)"
            
            return summary
        
        return "Emotion analysis method not implemented"
    
    def create_emotion_summary_for_analysis(self, results):
        """Create a concise emotion summary for AI analysis"""
        if not results:
            return "No emotion data available"
        
        emotions = [emotion for _, emotion, _ in results]
        confidences = [confidence for _, _, confidence in results]
        
        emotion_counts = Counter(emotions)
        avg_confidence = sum(confidences) / len(confidences)
        
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
    
    def extract_audio(self, video_path, output_folder="extracted_data"):
        """Extract audio from video file"""
        os.makedirs(output_folder, exist_ok=True)
        
        audio_path = os.path.join(output_folder, "audio.wav")
        try:
            self.log_to_console("Extracting audio from video...")
            video_clip = VideoFileClip(video_path)
            if video_clip.audio is None:
                self.log_to_console("Error: No audio track found in the video!")
                video_clip.close()
                return None
                
            video_clip.audio.write_audiofile(
                audio_path, 
                codec='pcm_s16le',
                fps=16000,
                verbose=False,
                logger=None
            )
            video_clip.close()
            
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                return audio_path
            else:
                self.log_to_console("Error: Audio file was not created or is empty.")
                return None
                
        except Exception as e:
            self.log_to_console(f"Error during audio extraction: {str(e)}")
            return None


def main():
    root = tk.Tk()
    app = StudentPitchAnalyzerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()