import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import sys
import os
import threading
from collections import Counter
from datetime import datetime
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
        self.root.title("🎯 Student Pitch Performance Analyzer")
        self.root.geometry("1400x900")
        
        # Modern color scheme
        self.colors = {
            'primary': '#2E86AB',      # Blue
            'secondary': '#A23B72',    # Purple
            'accent': '#F18F01',       # Orange
            'success': '#C73E1D',      # Red-orange
            'background': '#F5F7FA',   # Light gray
            'card': '#FFFFFF',         # White
            'text_primary': '#2D3748', # Dark gray
            'text_secondary': '#718096', # Medium gray
            'console_bg': '#1A202C',   # Dark
            'console_fg': '#E2E8F0'    # Light
        }
        
        self.root.configure(bg=self.colors['background'])
        
        # Variables
        self.video_path = tk.StringVar()
        self.analysis_results = {}
        self.is_analyzing = False
        self.start_time = None
        
        self.setup_styles()
        self.setup_ui()
        
    def setup_styles(self):
        """Configure custom styles for ttk widgets"""
        style = ttk.Style()
        
        # Configure notebook style
        style.configure('Custom.TNotebook', 
                       background=self.colors['card'],
                       borderwidth=0)
        style.configure('Custom.TNotebook.Tab',
                       padding=[20, 10],
                       font=('Segoe UI', 10, 'bold'))
        
        # Configure button styles
        style.configure('Primary.TButton',
                       background=self.colors['primary'],
                       foreground='white',
                       font=('Segoe UI', 11, 'bold'),
                       padding=[20, 10])
        
        style.configure('Secondary.TButton',
                       background=self.colors['secondary'],
                       foreground='white',
                       font=('Segoe UI', 10),
                       padding=[15, 8])
        
        style.configure('Accent.TButton',
                       background=self.colors['accent'],
                       foreground='white',
                       font=('Segoe UI', 12, 'bold'),
                       padding=[25, 12])
        
        # Configure frame styles
        style.configure('Card.TLabelFrame',
                       background=self.colors['card'],
                       relief='solid',
                       borderwidth=1)
        
        style.configure('Card.TLabelFrame.Label',
                       background=self.colors['card'],
                       foreground=self.colors['text_primary'],
                       font=('Segoe UI', 11, 'bold'))
        
    def setup_ui(self):
        # Main container with padding
        main_container = tk.Frame(self.root, bg=self.colors['background'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header section
        self.create_header(main_container)
        
        # File selection section
        self.create_file_section(main_container)
        
        # Progress and status section
        self.create_progress_section(main_container)
        
        # Main content area
        self.create_content_area(main_container)
        
        # Footer with action buttons
        self.create_footer(main_container)
        
    def create_header(self, parent):
        """Create the header section"""
        header_frame = tk.Frame(parent, bg=self.colors['background'], height=80)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        header_frame.pack_propagate(False)
        
        # Title with gradient-like effect using multiple labels
        title_frame = tk.Frame(header_frame, bg=self.colors['background'])
        title_frame.pack(expand=True)
        
        main_title = tk.Label(title_frame, text="🎯 Student Pitch Performance Analyzer",
                             font=('Segoe UI', 24, 'bold'),
                             fg=self.colors['primary'],
                             bg=self.colors['background'])
        main_title.pack()
        
        subtitle = tk.Label(title_frame, text="Advanced AI-Powered Analysis for Student Presentations",
                           font=('Segoe UI', 12),
                           fg=self.colors['text_secondary'],
                           bg=self.colors['background'])
        subtitle.pack(pady=(5, 0))
        
    def create_file_section(self, parent):
        """Create the file selection section"""
        file_frame = tk.Frame(parent, bg=self.colors['card'], relief='solid', bd=1)
        file_frame.pack(fill=tk.X, pady=(0, 15), ipady=20, ipadx=20)
        
        # Section title
        section_title = tk.Label(file_frame, text="📁 Video File Selection",
                                font=('Segoe UI', 14, 'bold'),
                                fg=self.colors['text_primary'],
                                bg=self.colors['card'])
        section_title.pack(anchor='w', pady=(0, 15))
        
        # File input row
        input_frame = tk.Frame(file_frame, bg=self.colors['card'])
        input_frame.pack(fill=tk.X)
        
        tk.Label(input_frame, text="Video File:",
                font=('Segoe UI', 11),
                fg=self.colors['text_secondary'],
                bg=self.colors['card']).pack(side=tk.LEFT, padx=(0, 10))
        
        self.file_entry = tk.Entry(input_frame, textvariable=self.video_path,
                                  font=('Segoe UI', 11),
                                  bg='white', fg=self.colors['text_primary'],
                                  relief='solid', bd=1)
        self.file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        browse_btn = tk.Button(input_frame, text="📂 Browse",
                              command=self.browse_file,
                              font=('Segoe UI', 10, 'bold'),
                              bg=self.colors['secondary'],
                              fg='white',
                              relief='flat',
                              padx=20, pady=8,
                              cursor='hand2')
        browse_btn.pack(side=tk.RIGHT)
        
        # Analyze button
        analyze_frame = tk.Frame(file_frame, bg=self.colors['card'])
        analyze_frame.pack(fill=tk.X, pady=(20, 0))
        
        self.analyze_button = tk.Button(analyze_frame, text="🚀 Start Analysis",
                                       command=self.start_analysis,
                                       font=('Segoe UI', 14, 'bold'),
                                       bg=self.colors['accent'],
                                       fg='white',
                                       relief='flat',
                                       padx=40, pady=12,
                                       cursor='hand2')
        self.analyze_button.pack()
        
    def create_progress_section(self, parent):
        """Create the progress and status section"""
        progress_frame = tk.Frame(parent, bg=self.colors['card'], relief='solid', bd=1)
        progress_frame.pack(fill=tk.X, pady=(0, 15), ipady=15, ipadx=20)
        
        # Status label
        self.status_label = tk.Label(progress_frame, text="Ready to analyze video",
                                    font=('Segoe UI', 11),
                                    fg=self.colors['text_secondary'],
                                    bg=self.colors['card'])
        self.status_label.pack(anchor='w', pady=(0, 10))
        
        # Progress bar
        self.progress = ttk.Progressbar(progress_frame, mode='indeterminate',
                                       style='TProgressbar')
        self.progress.pack(fill=tk.X, pady=(0, 10))
        
        # Time tracking
        self.time_label = tk.Label(progress_frame, text="",
                                  font=('Segoe UI', 9),
                                  fg=self.colors['text_secondary'],
                                  bg=self.colors['card'])
        self.time_label.pack(anchor='e')
        
    def create_content_area(self, parent):
        """Create the main content area with results and console"""
        content_frame = tk.Frame(parent, bg=self.colors['background'])
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Results section (left, 70%)
        results_frame = tk.Frame(content_frame, bg=self.colors['card'], relief='solid', bd=1)
        results_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        results_title = tk.Label(results_frame, text="📊 Analysis Results",
                                font=('Segoe UI', 14, 'bold'),
                                fg=self.colors['text_primary'],
                                bg=self.colors['card'])
        results_title.pack(anchor='w', padx=20, pady=(15, 10))
        
        # Notebook for different result tabs
        self.results_notebook = ttk.Notebook(results_frame, style='Custom.TNotebook')
        self.results_notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        # Create result tabs with enhanced styling
        self.create_result_tabs()
        
        # Console section (right, 30%)
        console_frame = tk.Frame(content_frame, bg=self.colors['console_bg'], relief='solid', bd=1)
        console_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(0, 0))
        console_frame.config(width=400)
        console_frame.pack_propagate(False)
        
        console_title = tk.Label(console_frame, text="💻 Console Output",
                                font=('Segoe UI', 12, 'bold'),
                                fg=self.colors['console_fg'],
                                bg=self.colors['console_bg'])
        console_title.pack(anchor='w', padx=15, pady=(15, 10))
        
        self.console_text = scrolledtext.ScrolledText(console_frame,
                                                     wrap=tk.WORD,
                                                     font=('Consolas', 9),
                                                     bg=self.colors['console_bg'],
                                                     fg=self.colors['console_fg'],
                                                     insertbackground=self.colors['console_fg'],
                                                     selectbackground=self.colors['primary'],
                                                     relief='flat',
                                                     padx=10, pady=10)
        self.console_text.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        
    def create_result_tabs(self):
        """Create enhanced result tabs"""
        tabs = [
            ("😊 Emotion Analysis", "emotion"),
            ("🎵 Audio Analysis", "audio"),
            ("📝 Transcription", "transcription"),
            ("🎯 Performance", "final")
        ]
        
        self.result_widgets = {}
        
        for tab_name, tab_key in tabs:
            frame = tk.Frame(self.results_notebook, bg='white')
            self.results_notebook.add(frame, text=tab_name)
            
            # Add padding frame
            padded_frame = tk.Frame(frame, bg='white')
            padded_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
            
            # Create text widget with custom styling
            text_widget = scrolledtext.ScrolledText(padded_frame,
                                                   wrap=tk.WORD,
                                                   font=('Segoe UI', 10),
                                                   relief='flat',
                                                   bg='white',
                                                   fg=self.colors['text_primary'],
                                                   selectbackground=self.colors['primary'])
            text_widget.pack(fill=tk.BOTH, expand=True)
            
            self.result_widgets[tab_key] = text_widget
        
    def create_footer(self, parent):
        """Create the footer with action buttons"""
        footer_frame = tk.Frame(parent, bg=self.colors['background'])
        footer_frame.pack(fill=tk.X)
        
        # Action buttons
        buttons_frame = tk.Frame(footer_frame, bg=self.colors['background'])
        buttons_frame.pack()
        
        buttons = [
            ("🗑️ Clear Results", self.clear_results, self.colors['text_secondary']),
            ("🧹 Clear Console", self.clear_console, self.colors['text_secondary']),
            ("💾 Save Results", self.save_results, self.colors['success']),
            ("📋 Export Report", self.export_detailed_report, self.colors['primary'])
        ]
        
        for text, command, color in buttons:
            btn = tk.Button(buttons_frame, text=text,
                           command=command,
                           font=('Segoe UI', 10, 'bold'),
                           bg=color,
                           fg='white',
                           relief='flat',
                           padx=20, pady=10,
                           cursor='hand2')
            btn.pack(side=tk.LEFT, padx=5)
    
    def browse_file(self):
        """Enhanced file browser with validation"""
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v"),
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("MOV files", "*.mov"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.video_path.set(filename)
            self.update_status(f"Selected: {os.path.basename(filename)}")
            
            # Show file info
            try:
                file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
                self.log_to_console(f"📁 File selected: {os.path.basename(filename)}")
                self.log_to_console(f"📏 File size: {file_size:.1f} MB")
            except Exception as e:
                self.log_to_console(f"⚠️ Error reading file info: {str(e)}")
    
    def update_status(self, message):
        """Update status label"""
        self.status_label.config(text=message)
        self.root.update_idletasks()
    
    def log_to_console(self, message):
        """Enhanced console logging with timestamps and colors"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        self.console_text.insert(tk.END, formatted_message)
        self.console_text.see(tk.END)
        self.root.update_idletasks()
    
    def clear_console(self):
        """Clear console with confirmation"""
        self.console_text.delete(1.0, tk.END)
        self.log_to_console("🧹 Console cleared")
    
    def clear_results(self):
        """Clear all results with confirmation"""
        if self.analysis_results:
            if messagebox.askyesno("Clear Results", "Are you sure you want to clear all analysis results?"):
                for widget in self.result_widgets.values():
                    widget.delete(1.0, tk.END)
                self.analysis_results = {}
                self.update_status("Results cleared")
                self.log_to_console("🗑️ Results cleared")
        else:
            messagebox.showinfo("No Results", "No results to clear.")
    
    def save_results(self):
        """Enhanced save functionality"""
        if not self.analysis_results:
            messagebox.showwarning("No Results", "No analysis results to save.")
            return
            
        filename = filedialog.asksaveasfilename(
            title="Save Analysis Results",
            defaultextension=".txt",
            filetypes=[
                ("Text files", "*.txt"),
                ("Markdown files", "*.md"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("Student Pitch Performance Analysis Results\n")
                    f.write("=" * 60 + "\n")
                    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Video file: {os.path.basename(self.video_path.get())}\n")
                    f.write("=" * 60 + "\n\n")
                    
                    for section, content in self.analysis_results.items():
                        f.write(f"{section}\n")
                        f.write("-" * len(section) + "\n")
                        f.write(str(content) + "\n\n")
                
                messagebox.showinfo("Success", f"Results saved to {filename}")
                self.log_to_console(f"💾 Results saved to {os.path.basename(filename)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {str(e)}")
                self.log_to_console(f"❌ Save failed: {str(e)}")
    
    def export_detailed_report(self):
        """Export detailed HTML report"""
        if not self.analysis_results:
            messagebox.showwarning("No Results", "No analysis results to export.")
            return
            
        filename = filedialog.asksaveasfilename(
            title="Export Detailed Report",
            defaultextension=".html",
            filetypes=[("HTML files", "*.html"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.create_html_report(filename)
                messagebox.showinfo("Success", f"Detailed report exported to {filename}")
                self.log_to_console(f"📋 Detailed report exported to {os.path.basename(filename)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export report: {str(e)}")
                self.log_to_console(f"❌ Export failed: {str(e)}")
    
    def create_html_report(self, filename):
        """Create a detailed HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Student Pitch Analysis Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background: #f5f7fa; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                h1 {{ color: {self.colors['primary']}; border-bottom: 3px solid {self.colors['primary']}; padding-bottom: 10px; }}
                h2 {{ color: {self.colors['secondary']}; margin-top: 30px; }}
                .meta {{ background: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }}
                .section {{ margin: 30px 0; padding: 20px; border-left: 4px solid {self.colors['accent']}; background: #fafbfc; }}
                pre {{ background: #2d3748; color: #e2e8f0; padding: 20px; border-radius: 5px; overflow-x: auto; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎯 Student Pitch Performance Analysis Report</h1>
                <div class="meta">
                    <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                    <strong>Video File:</strong> {os.path.basename(self.video_path.get())}<br>
                    <strong>Analysis Duration:</strong> {self.get_analysis_duration()}
                </div>
        """
        
        for section, content in self.analysis_results.items():
            html_content += f"""
                <div class="section">
                    <h2>{section}</h2>
                    <pre>{content}</pre>
                </div>
            """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def get_analysis_duration(self):
        """Get analysis duration"""
        if hasattr(self, 'analysis_start_time') and hasattr(self, 'analysis_end_time'):
            duration = self.analysis_end_time - self.analysis_start_time
            return f"{duration.total_seconds():.1f} seconds"
        return "Unknown"
    
    def start_analysis(self):
        """Enhanced analysis start with better validation"""
        if not self.video_path.get():
            messagebox.showerror("Error", "Please select a video file first.")
            return
        
        if not os.path.exists(self.video_path.get()):
            messagebox.showerror("Error", "Selected video file does not exist.")
            return
        
        if self.is_analyzing:
            messagebox.showwarning("Warning", "Analysis is already in progress.")
            return
        
        # Validate file format
        valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
        file_ext = os.path.splitext(self.video_path.get())[1].lower()
        if file_ext not in valid_extensions:
            if not messagebox.askyesno("Unknown Format", 
                                     f"The file format '{file_ext}' may not be supported. Continue anyway?"):
                return
        
        # Clear previous results
        self.clear_results()
        self.clear_console()
        
        # Start analysis
        self.is_analyzing = True
        self.analysis_start_time = datetime.now()
        self.analyze_button.config(state='disabled', text="🔄 Analyzing...", bg=self.colors['text_secondary'])
        self.progress.start(10)
        self.update_status("Starting analysis...")
        
        # Start timer update
        self.update_timer()
        
        thread = threading.Thread(target=self.run_analysis)
        thread.daemon = True
        thread.start()
    
    def update_timer(self):
        """Update analysis timer"""
        if self.is_analyzing and hasattr(self, 'analysis_start_time'):
            elapsed = datetime.now() - self.analysis_start_time
            self.time_label.config(text=f"Elapsed: {elapsed.total_seconds():.0f}s")
            self.root.after(1000, self.update_timer)
    
    def run_analysis(self):
        """Enhanced analysis pipeline with better error handling"""
        try:
            video_path = self.video_path.get()
            
            self.log_to_console("🚀 Starting comprehensive analysis pipeline...")
            self.update_status("Initializing analysis...")
            
            # Emotion Detection
            try:
                self.update_status("Analyzing facial emotions...")
                self.log_to_console("😊 Analyzing emotions in video...")
                results = detect_emotions_from_video(video_path)
                
                if results:
                    emotion_summary = self.summarize_emotion_data(results, method="detailed")
                    self.analysis_results["Emotion Analysis"] = emotion_summary
                    
                    self.root.after(0, lambda: self.result_widgets['emotion'].insert(tk.END, emotion_summary))
                    self.log_to_console("✅ Emotion detection completed successfully.")
                else:
                    error_msg = "⚠️ No emotions were detected in the video, or no faces were found."
                    self.analysis_results["Emotion Analysis"] = error_msg
                    self.root.after(0, lambda: self.result_widgets['emotion'].insert(tk.END, error_msg))
                    self.log_to_console(error_msg)
                    
            except Exception as e:
                error_msg = f"❌ Error during emotion detection: {str(e)}"
                self.log_to_console(error_msg)
                self.root.after(0, lambda: self.result_widgets['emotion'].insert(tk.END, error_msg))
            
            # Audio Extraction
            try:
                self.update_status("Extracting audio track...")
                self.log_to_console("🎵 Extracting audio from video...")
                audio_path = self.extract_audio(video_path)
                if audio_path:
                    self.log_to_console(f"✅ Audio extracted successfully to {audio_path}")
                else:
                    self.log_to_console("❌ No audio extracted due to an error.")
                    return
            except Exception as e:
                self.log_to_console(f"❌ Error during audio extraction: {str(e)}")
                return
            
            # Audio Analysis
            try:
                self.update_status("Analyzing audio characteristics...")
                self.log_to_console("🔊 Analyzing audio characteristics...")
                audio_result = analyze_audio(audio_path)
                if audio_result:
                    audio_text = self.format_audio_results(audio_result)
                    self.analysis_results["Audio Analysis"] = audio_text
                    self.root.after(0, lambda: self.result_widgets['audio'].insert(tk.END, audio_text))
                    self.log_to_console("✅ Audio analysis completed successfully.")
                else:
                    error_msg = "⚠️ No audio analysis results available."
                    self.analysis_results["Audio Analysis"] = error_msg
                    self.root.after(0, lambda: self.result_widgets['audio'].insert(tk.END, error_msg))
                    self.log_to_console(error_msg)
            except Exception as e:
                error_msg = f"❌ Error during audio analysis: {str(e)}"
                self.log_to_console(error_msg)
                self.root.after(0, lambda: self.result_widgets['audio'].insert(tk.END, error_msg))
                audio_result = None
            
            # Audio Transcription
            try:
                self.update_status("Transcribing speech...")
                self.log_to_console("📝 Transcribing audio...")
                audio_text = transcribe_audio(audio_path)
                if audio_text:
                    formatted_transcription = self.format_transcription(audio_text)
                    self.analysis_results["Transcription"] = formatted_transcription
                    self.root.after(0, lambda: self.result_widgets['transcription'].insert(tk.END, formatted_transcription))
                    self.log_to_console("✅ Audio transcription completed successfully.")
                else:
                    error_msg = "⚠️ No transcription result available."
                    self.analysis_results["Transcription"] = error_msg
                    self.root.after(0, lambda: self.result_widgets['transcription'].insert(tk.END, error_msg))
                    self.log_to_console(error_msg)
            except Exception as e:
                error_msg = f"❌ Error during audio transcription: {str(e)}"
                self.log_to_console(error_msg)
                self.root.after(0, lambda: self.result_widgets['transcription'].insert(tk.END, error_msg))
                audio_text = None
            
            # Final Analysis
            try:
                self.update_status("Generating performance insights...")
                self.log_to_console("🎯 Performing final pitch performance analysis...")
                if 'results' in locals():
                    emotion_summary = self.create_emotion_summary_for_analysis(results)
                else:
                    emotion_summary = "No emotion data available"
                
                final_result = analyze_student_pitch(emotion_summary, audio_result, audio_text)
                if final_result:
                    formatted_final = self.format_final_analysis(final_result)
                    self.analysis_results["Performance Analysis"] = formatted_final
                    self.root.after(0, lambda: self.result_widgets['final'].insert(tk.END, formatted_final))
                    self.log_to_console("✅ Student pitch performance analysis completed successfully.")
                else:
                    error_msg = "⚠️ No final analysis result available."
                    self.analysis_results["Performance Analysis"] = error_msg
                    self.root.after(0, lambda: self.result_widgets['final'].insert(tk.END, error_msg))
                    self.log_to_console(error_msg)
            except Exception as e:
                error_msg = f"❌ Error during final analysis: {str(e)}"
                self.log_to_console(error_msg)
                self.root.after(0, lambda: self.result_widgets['final'].insert(tk.END, error_msg))
            
            self.analysis_end_time = datetime.now()
            duration = self.analysis_end_time - self.analysis_start_time
            self.log_to_console(f"🎉 Analysis completed in {duration.total_seconds():.1f} seconds!")
            
        except Exception as e:
            error_msg = f"💥 Fatal error during analysis: {str(e)}"
            self.log_to_console(error_msg)
            messagebox.showerror("Error", error_msg)
        
        finally:
                self.is_analyzing = False
                self.root.after(0, self.reset_analysis_ui)
    
    def reset_analysis_ui(self):
        """Reset the analysis UI state"""
        self.analyze_button.config(state='normal', text="🚀 Start Analysis", bg=self.colors['accent'])
        self.progress.stop()
        self.update_status("Analysis completed")
    
    def extract_audio(self, video_path):
        """Extract audio from video file"""
        try:
            # Create audio output path
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            audio_path = os.path.join(os.path.dirname(video_path), f"{base_name}_audio.wav")
            
            # Suppress MoviePy output by redirecting stdout/stderr
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                # Extract audio using moviepy
                with VideoFileClip(video_path) as video:
                    if video.audio is not None:
                        video.audio.write_audiofile(audio_path, logger=None)
                        return audio_path
                    else:
                        self.log_to_console("⚠️ No audio track found in video")
                        return None
                    
        except Exception as e:
            self.log_to_console(f"❌ Error extracting audio: {str(e)}")
            return None
    
    def summarize_emotion_data(self, emotion_results, method="detailed"):
        """Create a comprehensive summary of emotion detection results"""
        if not emotion_results:
            return "No emotion data available."
        
        try:
            # Flatten all emotions from all frames
            all_emotions = []
            frame_count = 0
            
            for frame_data in emotion_results:
                frame_count += 1
                if 'emotions' in frame_data:
                    for emotion_dict in frame_data['emotions']:
                        if isinstance(emotion_dict, dict):
                            # Find the dominant emotion in this detection
                            if emotion_dict:
                                dominant_emotion = max(emotion_dict, key=emotion_dict.get)
                                all_emotions.append(dominant_emotion)
            
            if not all_emotions:
                return "No emotions detected in the analyzed frames."
            
            # Count emotions
            emotion_counts = Counter(all_emotions)
            total_detections = len(all_emotions)
            
            # Create detailed summary
            summary = []
            summary.append("🎭 EMOTION ANALYSIS SUMMARY")
            summary.append("=" * 50)
            summary.append(f"📊 Total frames analyzed: {frame_count}")
            summary.append(f"😊 Total emotion detections: {total_detections}")
            summary.append("")
            
            # Emotion distribution
            summary.append("📈 EMOTION DISTRIBUTION:")
            summary.append("-" * 30)
            for emotion, count in emotion_counts.most_common():
                percentage = (count / total_detections) * 100
                bar_length = int(percentage / 5)  # Scale for visual bar
                bar = "█" * bar_length + "░" * (20 - bar_length)
                summary.append(f"{emotion.capitalize():12} | {bar} | {count:3d} ({percentage:5.1f}%)")
            
            summary.append("")
            
            # Key insights
            summary.append("🔍 KEY INSIGHTS:")
            summary.append("-" * 20)
            
            if emotion_counts:
                dominant_emotion = emotion_counts.most_common(1)[0]
                summary.append(f"• Most prevalent emotion: {dominant_emotion[0].capitalize()} ({dominant_emotion[1]} detections)")
                
                if len(emotion_counts) > 1:
                    secondary_emotion = emotion_counts.most_common(2)[1]
                    summary.append(f"• Secondary emotion: {secondary_emotion[0].capitalize()} ({secondary_emotion[1]} detections)")
                
                # Emotional stability
                emotion_variety = len(emotion_counts)
                summary.append(f"• Emotional range: {emotion_variety} different emotions detected")
                
                if emotion_variety <= 2:
                    summary.append("• Emotional consistency: High (limited emotional range)")
                elif emotion_variety <= 4:
                    summary.append("• Emotional consistency: Moderate (balanced emotional expression)")
                else:
                    summary.append("• Emotional consistency: Variable (wide emotional range)")
            
            # Performance implications
            summary.append("")
            summary.append("🎯 PERFORMANCE IMPLICATIONS:")
            summary.append("-" * 30)
            
            positive_emotions = ['happy', 'joy', 'excited', 'confident']
            negative_emotions = ['sad', 'angry', 'fear', 'disgust', 'worried']
            neutral_emotions = ['neutral', 'calm', 'focused']
            
            positive_count = sum(count for emotion, count in emotion_counts.items() if emotion.lower() in positive_emotions)
            negative_count = sum(count for emotion, count in emotion_counts.items() if emotion.lower() in negative_emotions)
            neutral_count = sum(count for emotion, count in emotion_counts.items() if emotion.lower() in neutral_emotions)
            
            if positive_count > negative_count:
                summary.append("• Overall emotional tone: Positive")
                summary.append("• Likely to engage audience effectively")
            elif negative_count > positive_count:
                summary.append("• Overall emotional tone: Negative")
                summary.append("• May need to work on confidence and positivity")
            else:
                summary.append("• Overall emotional tone: Balanced")
                summary.append("• Good emotional control during presentation")
            
            return "\n".join(summary)
            
        except Exception as e:
            return f"Error processing emotion data: {str(e)}"
    
    def create_emotion_summary_for_analysis(self, emotion_results):
        """Create a brief emotion summary for the final analysis"""
        if not emotion_results:
            return "No emotion data available"
        
        try:
            all_emotions = []
            for frame_data in emotion_results:
                if 'emotions' in frame_data:
                    for emotion_dict in frame_data['emotions']:
                        if isinstance(emotion_dict, dict) and emotion_dict:
                            dominant_emotion = max(emotion_dict, key=emotion_dict.get)
                            all_emotions.append(dominant_emotion)
            
            if all_emotions:
                emotion_counts = Counter(all_emotions)
                dominant = emotion_counts.most_common(1)[0]
                return f"Dominant emotion: {dominant[0]} ({dominant[1]} detections), Total emotions: {len(all_emotions)}"
            else:
                return "No clear emotions detected"
                
        except Exception as e:
            return f"Error processing emotions: {str(e)}"
    
    def format_audio_results(self, audio_result):
        """Format audio analysis results for display"""
        if not audio_result:
            return "No audio analysis results available."
        
        try:
            formatted = []
            formatted.append("🎵 AUDIO ANALYSIS RESULTS")
            formatted.append("=" * 40)
            formatted.append("")
            
            if isinstance(audio_result, dict):
                for key, value in audio_result.items():
                    if isinstance(value, (int, float)):
                        formatted.append(f"{key.replace('_', ' ').title()}: {value:.2f}")
                    else:
                        formatted.append(f"{key.replace('_', ' ').title()}: {value}")
            else:
                formatted.append(str(audio_result))
            
            return "\n".join(formatted)
            
        except Exception as e:
            return f"Error formatting audio results: {str(e)}"
    
    def format_transcription(self, transcription_text):
        """Format transcription results for display"""
        if not transcription_text:
            return "No transcription available."
        
        try:
            formatted = []
            formatted.append("📝 SPEECH TRANSCRIPTION")
            formatted.append("=" * 30)
            formatted.append("")
            
            # Clean up the transcription
            cleaned_text = transcription_text.strip()
            
            # Add word count and reading time estimates
            word_count = len(cleaned_text.split())
            reading_time = word_count / 150  # Average reading speed
            
            formatted.append(f"📊 Word Count: {word_count}")
            formatted.append(f"⏱️ Estimated Reading Time: {reading_time:.1f} minutes")
            formatted.append(f"🗣️ Estimated Speaking Time: {word_count / 130:.1f} minutes")
            formatted.append("")
            formatted.append("📄 TRANSCRIBED CONTENT:")
            formatted.append("-" * 25)
            formatted.append(cleaned_text)
            
            return "\n".join(formatted)
            
        except Exception as e:
            return f"Error formatting transcription: {str(e)}"
    
    def format_final_analysis(self, final_result):
        """Format the final analysis results for display"""
        if not final_result:
            return "No final analysis available."
        
        try:
            formatted = []
            formatted.append("🎯 COMPREHENSIVE PERFORMANCE ANALYSIS")
            formatted.append("=" * 50)
            formatted.append("")
            
            # If the result is a string, display it directly
            if isinstance(final_result, str):
                formatted.append(final_result)
            else:
                # If it's a dict or other structure, format it nicely
                formatted.append(str(final_result))
            
            formatted.append("")
            formatted.append("🏆 ANALYSIS COMPLETE")
            formatted.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            return "\n".join(formatted)
            
        except Exception as e:
            return f"Error formatting final analysis: {str(e)}"


def main():
    """Main function to run the application"""
    try:
        root = tk.Tk()
        app = StudentPitchAnalyzerGUI(root)
        
        # Center the window on screen
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f'{width}x{height}+{x}+{y}')
        
        # Start the application
        root.mainloop()
        
    except Exception as e:
        print(f"Error starting application: {e}")
        messagebox.showerror("Startup Error", f"Failed to start application: {str(e)}")


if __name__ == "__main__":
    main()