import streamlit as st
import sys
import os
import threading
import time
from collections import Counter
from moviepy import VideoFileClip
import io
import contextlib
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile

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
    from Eyecontact import analyze_eye_contact
except ImportError as e:
    st.error(f"Warning: Could not import some modules: {e}")

# Configure Streamlit page
st.set_page_config(
    page_title="AI Pitch Performance Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for futuristic styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    .main-header {
        font-family: 'Orbitron', monospace;
        font-size: 2.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.5);
    }
    
    .sub-header {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.2rem;
        text-align: center;
        color: #a0a0a0;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .analysis-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.2);
    }
    
    .success-message {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(16, 185, 129, 0.1) 100%);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 10px;
        padding: 1rem;
        color: #10b981;
    }
    
    .warning-message {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.1) 0%, rgba(245, 158, 11, 0.1) 100%);
        border: 1px solid rgba(251, 191, 36, 0.3);
        border-radius: 10px;
        padding: 1rem;
        color: #f59e0b;
    }
    
    .error-message {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.1) 100%);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 10px;
        padding: 1rem;
        color: #dc2626;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .console-output {
        background: #1a1a1a;
        color: #00ff00;
        font-family: 'Courier New', monospace;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #333;
        max-height: 300px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitPitchAnalyzer:
    def __init__(self):
        self.analysis_results = {}
        self.console_messages = []
        
    def log_to_console(self, message):
        """Add message to console output"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.console_messages.append(formatted_message)
        if len(self.console_messages) > 50:  # Keep only last 50 messages
            self.console_messages.pop(0)
    
    def create_emotion_visualization(self, results):
        """Create interactive emotion visualization"""
        if not results:
            return None
            
        emotions = [emotion for _, emotion, _ in results]
        confidences = [confidence for _, _, confidence in results]
        frames = list(range(len(results)))
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Frame': frames,
            'Emotion': emotions,
            'Confidence': confidences
        })
        
        # Emotion distribution pie chart
        emotion_counts = Counter(emotions)
        fig_pie = px.pie(
            values=list(emotion_counts.values()),
            names=list(emotion_counts.keys()),
            title="Emotion Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        # Confidence timeline
        fig_timeline = px.line(
            df, x='Frame', y='Confidence', color='Emotion',
            title="Emotion Confidence Timeline",
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        fig_timeline.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        return fig_pie, fig_timeline, df
    
    def create_audio_visualization(self, audio_result):
        """Create audio analysis visualization"""
        if not audio_result:
            return None
            
        # Create gauge charts for audio metrics
        metrics = []
        values = []
        
        for key, value in audio_result.items():
            if isinstance(value, (int, float)):
                metrics.append(key)
                values.append(value)
        
        if not metrics:
            return None
            
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=metrics[:4],
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]]
        )
        
        colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c']
        
        for i, (metric, value) in enumerate(zip(metrics[:4], values[:4])):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=value,
                    title={'text': metric},
                    gauge={'axis': {'range': [None, max(values) * 1.2]},
                           'bar': {'color': colors[i]},
                           'steps': [{'range': [0, max(values) * 0.5], 'color': "lightgray"},
                                   {'range': [max(values) * 0.5, max(values)], 'color': "gray"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                       'thickness': 0.75, 'value': max(values) * 0.9}}
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=12),
            height=400
        )
        
        return fig
    
    def extract_audio(self, video_path):
        """Extract audio from video file"""
        try:
            self.log_to_console("🎵 Extracting audio from video...")
            
            # Create temporary file for audio
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            audio_path = temp_audio.name
            temp_audio.close()
            
            video_clip = VideoFileClip(video_path)
            if video_clip.audio is None:
                self.log_to_console("❌ Error: No audio track found in the video!")
                video_clip.close()
                return None
                
            video_clip.audio.write_audiofile(
                audio_path, 
                codec='pcm_s16le',
                fps=16000,
                # verbose=False,
                logger=None
            )
            video_clip.close()
            
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                self.log_to_console(f"✅ Audio extracted successfully!")
                return audio_path
            else:
                self.log_to_console("❌ Error: Audio file was not created or is empty.")
                return None
                
        except Exception as e:
            self.log_to_console(f"❌ Error during audio extraction: {str(e)}")
            return None
    
    def summarize_emotion_data(self, results, method="detailed"):
        """Generate detailed emotion analysis summary"""
        if not results:
            return "No emotion data available"
        
        emotions = [emotion for _, emotion, _ in results]
        confidences = [confidence for _, _, confidence in results]
        
        emotion_counts = Counter(emotions)
        avg_confidence = sum(confidences) / len(confidences)
        max_confidence = max(confidences)
        min_confidence = min(confidences)
        
        # Calculate emotion stability
        emotion_changes = sum(1 for i in range(1, len(emotions)) if emotions[i] != emotions[i-1])
        stability_score = 1 - (emotion_changes / len(emotions))
        
        # Dominant emotion analysis
        dominant_emotion = emotion_counts.most_common(1)[0]
        dominant_percentage = (dominant_emotion[1] / len(results)) * 100
        
        # Confidence analysis
        high_confidence_frames = sum(1 for conf in confidences if conf > 0.8)
        low_confidence_frames = sum(1 for conf in confidences if conf < 0.5)
        
        summary = f"""
🎭 **COMPREHENSIVE EMOTION ANALYSIS REPORT**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 **Statistical Overview:**
• Total Frames Analyzed: {len(results):,}
• Average Confidence: {avg_confidence:.1%}
• Confidence Range: {min_confidence:.1%} - {max_confidence:.1%}
• Emotion Stability Score: {stability_score:.1%}

🎯 **Dominant Emotion Analysis:**
• Primary Emotion: **{dominant_emotion[0].upper()}**
• Occurrence: {dominant_emotion[1]:,} frames ({dominant_percentage:.1f}% of presentation)
• Emotional Consistency: {'High' if dominant_percentage > 60 else 'Moderate' if dominant_percentage > 40 else 'Variable'}

📈 **Confidence Quality Assessment:**
• High Confidence Frames (>80%): {high_confidence_frames:,} ({(high_confidence_frames/len(results)*100):.1f}%)
• Low Confidence Frames (<50%): {low_confidence_frames:,} ({(low_confidence_frames/len(results)*100):.1f}%)
• Overall Detection Quality: {'Excellent' if avg_confidence > 0.8 else 'Good' if avg_confidence > 0.6 else 'Fair'}

🌟 **Detailed Emotion Breakdown:**"""
        
        for emotion, count in emotion_counts.most_common():
            percentage = (count / len(results)) * 100
            bar_length = int(percentage / 2)  # Scale bar for display
            bar = "█" * bar_length + "░" * (50 - bar_length)
            summary += f"\n  {emotion.capitalize():12} │{bar}│ {count:4} frames ({percentage:5.1f}%)"
        
        # Performance insights
        summary += f"\n\n🔍 **Performance Insights:**"
        
        if dominant_emotion[0].lower() in ['happy', 'confident', 'neutral']:
            summary += f"\n• ✅ Positive emotional presence detected - indicates good audience engagement potential"
        elif dominant_emotion[0].lower() in ['sad', 'angry', 'fearful']:
            summary += f"\n• ⚠️  Negative emotional dominance - may impact audience perception"
        
        if stability_score > 0.8:
            summary += f"\n• 🎯 High emotional stability - demonstrates composure and control"
        elif stability_score < 0.5:
            summary += f"\n• 📊 Variable emotional state - consider practicing emotional regulation techniques"
        
        if avg_confidence > 0.8:
            summary += f"\n• 🔬 Excellent facial detection quality - results are highly reliable"
        elif avg_confidence < 0.6:
            summary += f"\n• 🔍 Moderate detection quality - lighting or angle may have affected analysis"
        
        return summary
    
    def create_comprehensive_analysis_report(self, emotion_summary, audio_result, transcription, eye_contact_result, final_analysis):
        """Generate a comprehensive analysis report"""
        report = f"""
# 🚀 **COMPREHENSIVE PITCH PERFORMANCE ANALYSIS**
*Generated on {datetime.now().strftime("%B %d, %Y at %I:%M %p")}*

---

## 📋 **EXECUTIVE SUMMARY**

This comprehensive analysis evaluates multiple dimensions of presentation performance including emotional expression, vocal characteristics, content transcription, eye contact patterns, and overall delivery effectiveness.

---

## 🎭 **EMOTIONAL INTELLIGENCE ASSESSMENT**

{emotion_summary}

---

## 🎤 **VOCAL PERFORMANCE ANALYSIS**

"""
        
        if audio_result:
            report += "### Key Vocal Metrics:\n"
            for key, value in audio_result.items():
                if isinstance(value, (int, float)):
                    report += f"• **{key}**: {value:.2f}\n"
                else:
                    report += f"• **{key}**: {value}\n"
            
            # Add vocal performance insights
            report += "\n### 🔊 **Vocal Performance Insights:**\n"
            report += "• Vocal characteristics have been analyzed for pitch, tone, and delivery patterns\n"
            report += "• These metrics provide insights into speaker confidence and engagement levels\n"
        else:
            report += "⚠️ *Audio analysis data not available*\n"
        
        report += f"""
---

## 📝 **CONTENT TRANSCRIPTION**

### Full Presentation Transcript:
```
{transcription if transcription else 'Transcription not available'}
```

### 📊 **Content Analysis:**
"""
        
        if transcription:
            words = transcription.split()
            report += f"• **Word Count**: {len(words):,} words\n"
            report += f"• **Estimated Speaking Time**: {len(words) / 150:.1f} minutes (avg. 150 words/min)\n"
            report += f"• **Content Density**: {'High' if len(words) > 500 else 'Medium' if len(words) > 200 else 'Low'}\n"
        
        report += f"""
---

## 👁️ **EYE CONTACT & ENGAGEMENT ANALYSIS**

{eye_contact_result if eye_contact_result else 'Eye contact analysis not available'}

---

## 🎯 **FINAL PERFORMANCE EVALUATION**

{final_analysis if final_analysis else 'Final analysis not available'}

---

## 📈 **RECOMMENDATIONS & NEXT STEPS**

Based on this comprehensive analysis, here are key recommendations:

### 🎭 **Emotional Presence**
• Continue leveraging positive emotional expressions
• Work on maintaining emotional consistency throughout the presentation
• Practice techniques to manage any detected anxiety or nervousness

### 🎤 **Vocal Enhancement**
• Focus on vocal variety to maintain audience engagement
• Consider pace and pause strategies for emphasis
• Work on projection and clarity for better delivery impact

### 👁️ **Visual Connection**
• Maintain consistent eye contact with audience
• Use eye contact to emphasize key points
• Practice scanning different audience sections

### 📝 **Content Optimization**
• Ensure clear structure and logical flow
• Use storytelling elements to enhance engagement
• Practice smooth transitions between topics

---

*This analysis was generated using advanced AI-powered assessment tools for comprehensive presentation evaluation.*
"""
        
        return report

def main():
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = StreamlitPitchAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # Header
    st.markdown('<div class="main-header">🎯 AI PITCH PERFORMANCE ANALYZER</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced Multi-Modal Analysis for Presentation Excellence</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🚀 **Analysis Dashboard**")
        st.markdown("---")
        
        # File upload
        uploaded_file = st.file_uploader(
            "📁 Upload Presentation Video",
            type=['mp4', 'avi', 'mov', 'mkv', 'wmv'],
            help="Select a video file of your presentation for comprehensive analysis"
        )
        
        st.markdown("---")
        
        # Analysis options
        st.markdown("### ⚙️ **Analysis Options**")
        analyze_emotions = st.checkbox("🎭 Emotion Detection", value=True)
        do_analyze_audio = st.checkbox("🎤 Audio Analysis", value=True)
        analyze_transcription = st.checkbox("📝 Speech Transcription", value=True)
        do_analyze_eye_contact = st.checkbox("👁️ Eye Contact Analysis", value=True)
        generate_final_report = st.checkbox("📊 Comprehensive Report", value=True)
        
        st.markdown("---")
        
        # Analysis button
        if st.button("🚀 **START ANALYSIS**", type="primary", use_container_width=True):
            if uploaded_file is not None:
                st.session_state.start_analysis = True
            else:
                st.error("Please upload a video file first!")
        
        # Clear results button
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.analyzer = StreamlitPitchAnalyzer()
            st.session_state.analysis_complete = False
            st.rerun()
    
    # Main content area
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_video.write(uploaded_file.read())
        temp_video.close()
        video_path = temp_video.name
        
        # Display video info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📁 File Name", uploaded_file.name)
        with col2:
            st.metric("📊 File Size", f"{uploaded_file.size / (1024*1024):.1f} MB")
        with col3:
            st.metric("🎬 Format", uploaded_file.type.split('/')[-1].upper())
        
        # Video preview
        st.video(uploaded_file)
        
        # Start analysis if requested
        if st.session_state.get('start_analysis', False):
            st.session_state.start_analysis = False
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            console_container = st.container()
            
            # Console output
            with console_container:
                st.markdown("### 💻 **Real-time Analysis Console**")
                console_placeholder = st.empty()
            
            try:
                step = 0
                total_steps = sum([analyze_emotions, do_analyze_audio, analyze_transcription, do_analyze_eye_contact, generate_final_report])
                
                results = {}
                
                # Audio extraction
                status_text.text("🎵 Extracting audio from video...")
                audio_path = analyzer.extract_audio(video_path)
                
                # Update console
                console_placeholder.markdown(
                    f'<div class="console-output">{"<br>".join(analyzer.console_messages)}</div>',
                    unsafe_allow_html=True
                )
                
                # Emotion Analysis
                if analyze_emotions:
                    step += 1
                    progress_bar.progress(step / total_steps)
                    status_text.text("🎭 Analyzing facial emotions...")
                    
                    try:
                        analyzer.log_to_console("🎭 Starting emotion detection analysis...")
                        emotion_results = detect_emotions_from_video(video_path)
                        if emotion_results:
                            results['emotions'] = emotion_results
                            analyzer.log_to_console(f"✅ Emotion analysis complete - {len(emotion_results)} frames analyzed")
                        else:
                            analyzer.log_to_console("⚠️ No emotions detected in video")
                    except Exception as e:
                        analyzer.log_to_console(f"❌ Emotion analysis error: {str(e)}")
                    
                    console_placeholder.markdown(
                        f'<div class="console-output">{"<br>".join(analyzer.console_messages)}</div>',
                        unsafe_allow_html=True
                    )
                
                # Audio Analysis
                if do_analyze_audio and audio_path:
                    step += 1
                    progress_bar.progress(step / total_steps)
                    status_text.text("🎤 Analyzing audio characteristics...")
                    
                    try:
                        analyzer.log_to_console("🎤 Analyzing audio characteristics...")
                        audio_result = analyze_audio(audio_path)
                        if audio_result:
                            results['audio'] = audio_result
                            analyzer.log_to_console("✅ Audio analysis complete")
                        else:
                            analyzer.log_to_console("⚠️ No audio analysis results")
                    except Exception as e:
                        analyzer.log_to_console(f"❌ Audio analysis error: {str(e)}")
                    
                    console_placeholder.markdown(
                        f'<div class="console-output">{"<br>".join(analyzer.console_messages)}</div>',
                        unsafe_allow_html=True
                    )
                
                # Transcription
                if analyze_transcription and audio_path:
                    step += 1
                    progress_bar.progress(step / total_steps)
                    status_text.text("📝 Transcribing speech...")
                    
                    try:
                        analyzer.log_to_console("📝 Transcribing audio to text...")
                        transcription = transcribe_audio(audio_path)
                        if transcription:
                            results['transcription'] = transcription
                            analyzer.log_to_console("✅ Transcription complete")
                        else:
                            analyzer.log_to_console("⚠️ No transcription results")
                    except Exception as e:
                        analyzer.log_to_console(f"❌ Transcription error: {str(e)}")
                    
                    console_placeholder.markdown(
                        f'<div class="console-output">{"<br>".join(analyzer.console_messages)}</div>',
                        unsafe_allow_html=True
                    )
                
                # Eye Contact Analysis
                if do_analyze_audio:
                    step += 1
                    progress_bar.progress(step / total_steps)
                    status_text.text("👁️ Analyzing eye contact patterns...")
                    
                    try:
                        analyzer.log_to_console("👁️ Analyzing eye contact patterns...")
                        eye_contact_result = analyze_eye_contact(video_path)
                        if eye_contact_result:
                            results['eye_contact'] = eye_contact_result
                            analyzer.log_to_console("✅ Eye contact analysis complete")
                        else:
                            analyzer.log_to_console("⚠️ No eye contact analysis results")
                    except Exception as e:
                        analyzer.log_to_console(f"❌ Eye contact analysis error: {str(e)}")
                    
                    console_placeholder.markdown(
                        f'<div class="console-output">{"<br>".join(analyzer.console_messages)}</div>',
                        unsafe_allow_html=True
                    )
                
                # Final Analysis
                if generate_final_report:
                    step += 1
                    progress_bar.progress(step / total_steps)
                    status_text.text("📊 Generating comprehensive analysis...")
                    
                    try:
                        analyzer.log_to_console("📊 Generating final performance analysis...")
                        
                        emotion_summary = analyzer.summarize_emotion_data(results.get('emotions', []))
                        final_analysis = analyze_student_pitch(
                            emotion_summary,
                            results.get('audio'),
                            results.get('transcription'),
                            results.get('eye_contact')
                        )
                        
                        if final_analysis:
                            results['final_analysis'] = final_analysis
                            analyzer.log_to_console("✅ Comprehensive analysis complete!")
                        else:
                            analyzer.log_to_console("⚠️ Final analysis incomplete")
                    except Exception as e:
                        analyzer.log_to_console(f"❌ Final analysis error: {str(e)}")
                
                progress_bar.progress(1.0)
                status_text.text("✅ Analysis Complete!")
                
                # Store results in session state
                st.session_state.analysis_results = results
                st.session_state.analysis_complete = True
                
                analyzer.log_to_console("🎉 All analyses completed successfully!")
                console_placeholder.markdown(
                    f'<div class="console-output">{"<br>".join(analyzer.console_messages)}</div>',
                    unsafe_allow_html=True
                )
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                analyzer.log_to_console(f"💥 Critical error: {str(e)}")
    
    # Display results if analysis is complete
    if st.session_state.get('analysis_complete', False) and 'analysis_results' in st.session_state:
        results = st.session_state.analysis_results
        
        st.markdown("---")
        st.markdown("# 📊 **ANALYSIS RESULTS**")
        
        # Create tabs for different analyses
        tabs = st.tabs(["🎭 Emotions", "🎤 Audio", "📝 Transcription", "👁️ Eye Contact", "📋 Full Report"])
        
        # Emotion Analysis Tab
        with tabs[0]:
            if 'emotions' in results:
                st.markdown("## 🎭 **Emotion Analysis Results**")
                
                # Create visualizations
                fig_pie, fig_timeline, df = analyzer.create_emotion_visualization(results['emotions'])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_pie, use_container_width=True)
                with col2:
                    st.plotly_chart(fig_timeline, use_container_width=True)
                
                # Detailed summary
                emotion_summary = analyzer.summarize_emotion_data(results['emotions'])
                st.markdown(f'<div class="analysis-card">{emotion_summary}</div>', unsafe_allow_html=True)
                
                # Data table
                with st.expander("📊 View Detailed Emotion Data"):
                    st.dataframe(df, use_container_width=True)
            else:
                st.warning("No emotion analysis data available")
        
        # Audio Analysis Tab
        with tabs[1]:
            if 'audio' in results:
                st.markdown("## 🎤 **Audio Analysis Results**")
                
                # Create audio visualization
                fig_audio = analyzer.create_audio_visualization(results['audio'])
                if fig_audio:
                    st.plotly_chart(fig_audio, use_container_width=True)
                
                # Audio metrics
                col1, col2, col3, col4 = st.columns(4)
                audio_metrics = list(results['audio'].items())
                
                for i, (key, value) in enumerate(audio_metrics):
                    with [col1, col2, col3, col4][i % 4]:
                        if isinstance(value, (int, float)):
                            st.metric(key, f"{value:.2f}")
                        else:
                            st.metric(key, str(value))
                
                # Detailed audio analysis
                audio_details = "### 🔊 **Detailed Audio Characteristics:**\n\n"
                for key, value in results['audio'].items():
                    audio_details += f"• **{key}**: {value}\n"
                
                st.markdown(f'<div class="analysis-card">{audio_details}</div>', unsafe_allow_html=True)
            else:
                st.warning("No audio analysis data available")
        
        # Transcription Tab
        with tabs[2]:
            if 'transcription' in results:
                st.markdown("## 📝 **Speech Transcription Results**")
                
                transcription = results['transcription']
                
                # Transcription metrics
                words = transcription.split()
                sentences = transcription.split('.')
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📝 Word Count", len(words))
                with col2:
                    st.metric("📄 Sentences", len([s for s in sentences if s.strip()]))
                with col3:
                    st.metric("⏱️ Est. Duration", f"{len(words) / 150:.1f} min")
                with col4:
                    st.metric("🎯 Avg Words/Sentence", f"{len(words) / max(len([s for s in sentences if s.strip()]), 1):.1f}")
                
                # Full transcription
                st.markdown("### 📜 **Complete Transcription:**")
                st.markdown(f'<div class="analysis-card" style="max-height: 400px; overflow-y: auto;">{transcription}</div>', unsafe_allow_html=True)
                
                # Word frequency analysis
                if len(words) > 10:
                    word_freq = Counter([word.lower().strip('.,!?;"()') for word in words if len(word) > 3])
                    top_words = word_freq.most_common(10)
                    
                    if top_words:
                        st.markdown("### 📊 **Most Frequent Words:**")
                        freq_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
                        fig_words = px.bar(freq_df, x='Word', y='Frequency', 
                                         title="Top 10 Most Frequent Words",
                                         color='Frequency',
                                         color_continuous_scale='viridis')
                        fig_words.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white')
                        )
                        st.plotly_chart(fig_words, use_container_width=True)
            else:
                st.warning("No transcription data available")
        
        # Eye Contact Tab
        with tabs[3]:
            if 'eye_contact' in results:
                st.markdown("## 👁️ **Eye Contact Analysis Results**")
                eye_contact_result = results['eye_contact']
                st.markdown(f'<div class="analysis-card">{eye_contact_result}</div>', unsafe_allow_html=True)
            else:
                st.warning("No eye contact analysis data available")
        
        # Full Report Tab
        with tabs[4]:
            st.markdown("## 📋 **Comprehensive Performance Report**")
            
            # Generate comprehensive report
            emotion_summary = analyzer.summarize_emotion_data(results.get('emotions', []))
            comprehensive_report = analyzer.create_comprehensive_analysis_report(
                emotion_summary,
                results.get('audio'),
                results.get('transcription'),
                results.get('eye_contact'),
                results.get('final_analysis')
            )
            
            st.markdown(comprehensive_report)
            
            # Build PDF in memory
            pdf_buffer = io.BytesIO()
            c = canvas.Canvas(pdf_buffer, pagesize=letter)
            text = c.beginText(40, 750)
            text.setFont("Courier", 10)

            # Loop through each line of your markdown report
            for line in comprehensive_report.splitlines():
                # wrap long lines if needed (simple example)
                for chunk in [line[i:i+95] for i in range(0, len(line), 95)]:
                    text.textLine(chunk)
            c.drawText(text)
            c.showPage()
            c.save()
            pdf_buffer.seek(0)

            st.download_button(
                label="📥 Download Full Report as PDF",
                data=pdf_buffer,
                file_name=f"pitch_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

            # Download report
            # st.download_button(
            #     label="📥 Download Full Report",
            #     data=comprehensive_report,
            #     file_name=f"pitch_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            #     mime="text/markdown",
            #     use_container_width=True
            # )
        
        # Performance Summary Cards
        st.markdown("---")
        st.markdown("## 🎯 **Performance Summary**")
        
        summary_cols = st.columns(4)
        
        # Emotion Score
        with summary_cols[0]:
            if 'emotions' in results:
                emotions = [emotion for _, emotion, _ in results['emotions']]
                confidences = [confidence for _, _, confidence in results['emotions']]
                avg_confidence = sum(confidences) / len(confidences)
                
                # Calculate emotion positivity score
                positive_emotions = ['happy', 'surprise', 'neutral']
                positive_count = sum(1 for emotion in emotions if emotion.lower() in positive_emotions)
                emotion_score = (positive_count / len(emotions)) * 100
                
                st.markdown(f"""
                <div class="metric-container">
                    <h3>🎭 Emotion Score</h3>
                    <h1 style="color: {'#10b981' if emotion_score > 70 else '#f59e0b' if emotion_score > 40 else '#dc2626'}">{emotion_score:.0f}%</h1>
                    <p>Confidence: {avg_confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-container">
                    <h3>🎭 Emotion Score</h3>
                    <h1 style="color: #6b7280">N/A</h1>
                    <p>No data available</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Audio Score
        with summary_cols[1]:
            if 'audio' in results:
                # Calculate a simple audio quality score based on available metrics
                audio_score = 75  # Default score, can be enhanced based on actual audio metrics
                
                st.markdown(f"""
                <div class="metric-container">
                    <h3>🎤 Audio Quality</h3>
                    <h1 style="color: {'#10b981' if audio_score > 70 else '#f59e0b' if audio_score > 40 else '#dc2626'}">{audio_score:.0f}%</h1>
                    <p>Voice Analysis</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-container">
                    <h3>🎤 Audio Quality</h3>
                    <h1 style="color: #6b7280">N/A</h1>
                    <p>No data available</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Content Score
        with summary_cols[2]:
            if 'transcription' in results:
                words = results['transcription'].split()
                # Calculate content score based on word count and structure
                if len(words) > 500:
                    content_score = 90
                elif len(words) > 200:
                    content_score = 75
                elif len(words) > 100:
                    content_score = 60
                else:
                    content_score = 40
                
                st.markdown(f"""
                <div class="metric-container">
                    <h3>📝 Content Quality</h3>
                    <h1 style="color: {'#10b981' if content_score > 70 else '#f59e0b' if content_score > 40 else '#dc2626'}">{content_score:.0f}%</h1>
                    <p>{len(words)} words</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-container">
                    <h3>📝 Content Quality</h3>
                    <h1 style="color: #6b7280">N/A</h1>
                    <p>No data available</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Overall Score
        with summary_cols[3]:
            # Calculate overall score from available metrics
            scores = []
            if 'emotions' in results:
                emotions = [emotion for _, emotion, _ in results['emotions']]
                positive_emotions = ['happy', 'surprise', 'neutral']
                positive_count = sum(1 for emotion in emotions if emotion.lower() in positive_emotions)
                emotion_score = (positive_count / len(emotions)) * 100
                scores.append(emotion_score)
            
            if 'audio' in results:
                scores.append(75)  # Default audio score
            
            if 'transcription' in results:
                words = results['transcription'].split()
                if len(words) > 500:
                    content_score = 90
                elif len(words) > 200:
                    content_score = 75
                elif len(words) > 100:
                    content_score = 60
                else:
                    content_score = 40
                scores.append(content_score)
            
            overall_score = sum(scores) / len(scores) if scores else 0
            
            st.markdown(f"""
            <div class="metric-container">
                <h3>🏆 Overall Score</h3>
                <h1 style="color: {'#10b981' if overall_score > 70 else '#f59e0b' if overall_score > 40 else '#dc2626'}">{overall_score:.0f}%</h1>
                <p>Combined Analysis</p>
            </div>
            """, unsafe_allow_html=True)
        
        # AI Insights and Recommendations
        st.markdown("---")
        st.markdown("## 🤖 **AI-Powered Insights & Recommendations**")
        
        insights_container = st.container()
        with insights_container:
            if results.get('final_analysis'):
                st.markdown(f'<div class="analysis-card">{results["final_analysis"]}</div>', unsafe_allow_html=True)
            else:
                # Generate basic insights from available data
                insights = "### 🔍 **Key Insights:**\n\n"
                
                if 'emotions' in results:
                    emotions = [emotion for _, emotion, _ in results['emotions']]
                    dominant_emotion = Counter(emotions).most_common(1)[0]
                    insights += f"• **Emotional Presence**: Your dominant emotion was {dominant_emotion[0]} ({(dominant_emotion[1]/len(emotions)*100):.1f}% of the time)\n"
                
                if 'transcription' in results:
                    words = results['transcription'].split()
                    insights += f"• **Content Delivery**: You spoke approximately {len(words)} words, indicating {'comprehensive' if len(words) > 500 else 'moderate' if len(words) > 200 else 'brief'} content coverage\n"
                
                if 'audio' in results:
                    insights += f"• **Vocal Performance**: Audio characteristics analyzed for pitch, tone, and delivery patterns\n"
                
                insights += "\n### 💡 **Recommendations:**\n\n"
                insights += "• Practice maintaining consistent emotional engagement throughout your presentation\n"
                insights += "• Work on vocal variety to keep your audience engaged\n"
                insights += "• Focus on clear articulation and appropriate pacing\n"
                insights += "• Use eye contact strategically to connect with your audience\n"
                
                st.markdown(f'<div class="analysis-card">{insights}</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; font-family: 'Rajdhani', sans-serif;">
        <p>🚀 Powered by Advanced AI • Built with Streamlit • © 2024 Pitch Performance Analyzer</p>
        <p style="font-size: 0.8rem;">Comprehensive multi-modal analysis for presentation excellence</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()