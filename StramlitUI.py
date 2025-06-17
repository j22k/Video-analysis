# app.py (Previously StramlitUI.py)
import streamlit as st

# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(
    page_title="AI Pitch Performance Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Standard Library Imports ---
import os
import io
import tempfile
from collections import Counter
from datetime import datetime
import pandas as pd
import numpy as np

# --- Third-Party Library Imports ---
try:
    from moviepy import VideoFileClip
except ImportError:
    st.error("MoviePy not found. Please install it: pip install moviepy")
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
except ImportError:
    st.error("ReportLab not found. Please install it: pip install reportlab")

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Custom Module Imports & Health Check ---
MODULE_HEALTH = {
    "emotion": False, "transcription": False, "llm_analysis": False,
    "audio": False, "eye_contact": False
}
MODULE_ERRORS = {}

try:
    from emotion_detection.EmotionDetection import detect_emotions_from_video
    MODULE_HEALTH["emotion"] = True
except ImportError as e:
    MODULE_ERRORS["emotion"] = f"Could not import EmotionDetection module. Error: {e}"

try:
    from transcribe_audio import transcribe_audio
    MODULE_HEALTH["transcription"] = True
except ImportError as e:
    MODULE_ERRORS["transcription"] = f"Could not import transcription module. Error: {e}"

try:
    from lanchain_deepseek import analyze_student_pitch
    MODULE_HEALTH["llm_analysis"] = True
except ImportError as e:
    MODULE_ERRORS["llm_analysis"] = f"Could not import LLM analysis module. Error: {e}"

try:
    from audio_analysis import analyze_audio
    MODULE_HEALTH["audio"] = True
except ImportError as e:
    MODULE_ERRORS["audio"] = f"Could not import audio analysis module. Error: {e}"

try:
    from eye_contact import analyze_eye_contact
    MODULE_HEALTH["eye_contact"] = True
except ImportError as e:
    MODULE_ERRORS["eye_contact"] = f"Could not import eye contact module. Error: {e}"


# --- Custom CSS for Styling ---
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
        font-size: 0.85em;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)


class StreamlitPitchAnalyzer:
    def __init__(self):
        self.console_messages = []

    def log_to_console(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console_messages.append(f"[{timestamp}] {message}")
        if len(self.console_messages) > 100:
            self.console_messages.pop(0)

    def create_emotion_visualization(self, results):
        if not results: return None, None, None
        df = pd.DataFrame(results, columns=['Frame', 'Emotion', 'Confidence'])
        
        emotion_counts = df['Emotion'].value_counts()
        fig_pie = px.pie(
            values=emotion_counts.values,
            names=emotion_counts.index,
            title="Emotion Distribution"
        )
        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        
        fig_timeline = px.line(
            df, x='Frame', y='Confidence', color='Emotion',
            title="Emotion Confidence Timeline",
            labels={'Confidence': 'Confidence Level', 'Frame': 'Video Frame'}
        )
        fig_timeline.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        
        return fig_pie, fig_timeline, df

    def create_audio_visualization(self, audio_result):
        if not audio_result: return None
        metrics_to_plot = {k: v for k, v in audio_result.items() if isinstance(v, (int, float))}
        if not metrics_to_plot: return None
        
        fig = make_subplots(
            rows=1, cols=len(metrics_to_plot),
            specs=[[{"type": "indicator"}] * len(metrics_to_plot)],
            subplot_titles=list(metrics_to_plot.keys())
        )
        
        colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c']
        for i, (name, value) in enumerate(metrics_to_plot.items()):
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=value,
                title={'text': name, 'font': {'size': 14}},
                gauge={'axis': {'range': [None, value * 1.5]},
                       'bar': {'color': colors[i % len(colors)]}}
            ), row=1, col=i+1)
            
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=250,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        return fig

    def extract_audio(self, video_path):
        try:
            self.log_to_console("🎵 Extracting audio from video...")
            with VideoFileClip(video_path) as video_clip:
                if video_clip.audio is None:
                    self.log_to_console("❌ Error: No audio track found in the video!")
                    return None
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                    audio_path = temp_audio.name
                
                video_clip.audio.write_audiofile(
                    audio_path, codec='pcm_s16le', fps=16000, logger=None
                )
            
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                self.log_to_console("✅ Audio extracted successfully!")
                return audio_path
            else:
                self.log_to_console("❌ Error: Audio file was not created or is empty.")
                return None
        except Exception as e:
            self.log_to_console(f"❌ Critical error during audio extraction: {e}")
            return None

    def summarize_emotion_data(self, results):
        if not results: return "No emotion data available."
        df = pd.DataFrame(results, columns=['Frame', 'Emotion', 'Confidence'])
        
        emotion_counts = df['Emotion'].value_counts(normalize=True) * 100
        avg_confidence = df['Confidence'].mean()
        
        summary = "#### 🎭 Emotion Summary\n"
        summary += f"- **Dominant Emotion**: {emotion_counts.index[0]} ({emotion_counts.iloc[0]:.1f}%)\n"
        summary += f"- **Average Confidence**: {avg_confidence:.1%}\n"
        summary += "- **Emotion Distribution**:\n"
        for emotion, percentage in emotion_counts.items():
            summary += f"  - {emotion.capitalize()}: {percentage:.1f}%\n"
        return summary
        
    def create_pdf_report(self, report_data):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        lines = report_data.split('\n')
        for line in lines:
            if line.startswith('# '):
                story.append(Paragraph(line.replace('# ', ''), styles['h1']))
            elif line.startswith('## '):
                story.append(Paragraph(line.replace('## ', ''), styles['h2']))
            elif line.startswith('### '):
                story.append(Paragraph(line.replace('### ', ''), styles['h3']))
            elif line.strip().startswith('• '):
                 story.append(Paragraph(line.strip(), styles['Bullet']))
            elif line.strip() == '---':
                 story.append(Spacer(1, 12))
            else:
                story.append(Paragraph(line, styles['BodyText']))
        
        doc.build(story)
        buffer.seek(0)
        return buffer

    def create_comprehensive_analysis_report(self, results):
        emotion_summary = self.summarize_emotion_data(results.get('emotions', []))
        audio_result = results.get('audio')
        transcription = results.get('transcription')
        eye_contact_result = results.get('eye_contact')
        final_analysis = results.get('final_analysis')

        report = f"""
# 🚀 Comprehensive Pitch Performance Analysis
*Generated on {datetime.now().strftime("%B %d, %Y at %I:%M %p")}*
---
## 🎭 Emotional Intelligence Assessment
{emotion_summary}
---
## 🎤 Vocal Performance Analysis
"""
        if audio_result:
            # <<< FIX: Robustly handle mixed data types (numbers and strings)
            for key, value in audio_result.items():
                if isinstance(value, (int, float)):
                    # Format numbers to 2 decimal places
                    formatted_value = f"{value:.2f}"
                else:
                    # For anything else (like strings), just use its string representation
                    formatted_value = str(value)
                
                # Safely add the formatted line to the report
                report += f"• **{key.replace('_', ' ').title()}**: {formatted_value}\n"
        else:
            report += "⚠️ *Audio analysis data not available.*\n"
        
        report += f"""
---
## 📝 Content & Transcription Analysis
**Full Transcript:**
{transcription if transcription else 'Transcription not available.'}
"""
        if transcription:
            words = transcription.split()
            report += f"• **Word Count**: {len(words):,} words\n"
        report += f"""
---
## 👁️ Eye Contact & Engagement Analysis
{eye_contact_result if eye_contact_result else 'Eye contact analysis not available.'}
---
## 🎯 Final AI-Powered Evaluation
{final_analysis if final_analysis else 'Final LLM-based analysis not available.'}
---
"""
        return report


def main():
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = StreamlitPitchAnalyzer()
    analyzer = st.session_state.analyzer

    st.markdown('<div class="main-header">🎯 AI PITCH PERFORMANCE ANALYZER</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced Multi-Modal Analysis for Presentation Excellence</div>', unsafe_allow_html=True)

    if MODULE_ERRORS:
        for key, error_msg in MODULE_ERRORS.items():
            st.warning(f"**Module Load Failure ({key}):** {error_msg}")
        st.info("Some analysis options may be disabled. Please check your installation and file structure.")

    with st.sidebar:
        st.markdown("## 🚀 **Analysis Dashboard**")
        st.markdown("---")
        
        uploaded_file = st.file_uploader("📁 Upload Presentation Video", type=['mp4', 'mov', 'avi', 'mkv'])
        
        st.markdown("---")
        st.markdown("### ⚙️ **Analysis Options**")
        
        analyze_emotions = st.checkbox("🎭 Emotion Detection", value=True, disabled=not MODULE_HEALTH["emotion"], help=MODULE_ERRORS.get("emotion"))
        do_analyze_audio = st.checkbox("🎤 Audio Analysis", value=True, disabled=not MODULE_HEALTH["audio"], help=MODULE_ERRORS.get("audio"))
        analyze_transcription = st.checkbox("📝 Speech Transcription", value=True, disabled=not MODULE_HEALTH["transcription"], help=MODULE_ERRORS.get("transcription"))
        do_analyze_eye_contact = st.checkbox("👁️ Eye Contact Analysis", value=True, disabled=not MODULE_HEALTH["eye_contact"], help=MODULE_ERRORS.get("eye_contact"))
        generate_final_report = st.checkbox("📊 Comprehensive LLM Report", value=True, disabled=not MODULE_HEALTH["llm_analysis"], help=MODULE_ERRORS.get("llm_analysis"))
        
        st.markdown("---")
        
        if st.button("🚀 **START ANALYSIS**", type="primary", use_container_width=True):
            if uploaded_file is not None:
                st.session_state.analysis_results = {}
                st.session_state.analysis_complete = False
                st.session_state.start_analysis = True
                st.rerun()
            else:
                st.error("Please upload a video file first!")
        
        if st.button("🗑️ Clear Results", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key not in ['analyzer']:
                    del st.session_state[key]
            st.rerun()

    if not uploaded_file and not st.session_state.get('analysis_complete'):
        st.info("Upload a video and click 'Start Analysis' to begin.")
        return

    if st.session_state.get('start_analysis'):
        video_path = None
        audio_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_video:
                temp_video.write(uploaded_file.getvalue())
                video_path = temp_video.name

            st.video(video_path)
            
            progress_bar = st.progress(0, "Initializing Analysis...")
            status_text = st.empty()
            console_container = st.expander("💻 Real-time Analysis Console", expanded=True)
            console_placeholder = console_container.empty()

            analysis_tasks = {
                "emotions": analyze_emotions, "audio": do_analyze_audio,
                "transcription": analyze_transcription, "eye_contact": do_analyze_eye_contact,
                "final_report": generate_final_report
            }
            tasks_to_run = [k for k, v in analysis_tasks.items() if v]
            total_steps = len(tasks_to_run) + 1 
            step = 0
            
            results = {}

            if do_analyze_audio or analyze_transcription:
                audio_path = analyzer.extract_audio(video_path)
            step += 1
            progress_bar.progress(step / total_steps, "Audio Extracted")
            console_placeholder.markdown(f'<div class="console-output">{"<br>".join(analyzer.console_messages)}</div>', unsafe_allow_html=True)
            
            if analyze_emotions:
                status_text.text("🎭 Analyzing facial emotions...")
                try:
                    analyzer.log_to_console("Running emotion detection...")
                    results['emotions'] = detect_emotions_from_video(video_path)
                    analyzer.log_to_console("✅ Emotion analysis complete.")
                except Exception as e:
                    analyzer.log_to_console(f"❌ Emotion analysis failed: {e}")
                step += 1
                progress_bar.progress(step / total_steps, "Emotions Analyzed")
                console_placeholder.markdown(f'<div class="console-output">{"<br>".join(analyzer.console_messages)}</div>', unsafe_allow_html=True)
            
            if do_analyze_audio and audio_path:
                status_text.text("🎤 Analyzing audio characteristics...")
                try:
                    analyzer.log_to_console("Running audio analysis...")
                    results['audio'] = analyze_audio(audio_path)
                    analyzer.log_to_console("✅ Audio analysis complete.")
                except Exception as e:
                    analyzer.log_to_console(f"❌ Audio analysis failed: {e}")
                step += 1
                progress_bar.progress(step / total_steps, "Audio Analyzed")
                console_placeholder.markdown(f'<div class="console-output">{"<br>".join(analyzer.console_messages)}</div>', unsafe_allow_html=True)

            if analyze_transcription and audio_path:
                status_text.text("📝 Transcribing speech...")
                try:
                    analyzer.log_to_console("Running speech-to-text...")
                    results['transcription'] = transcribe_audio(audio_path)
                    analyzer.log_to_console("✅ Transcription complete.")
                except Exception as e:
                    analyzer.log_to_console(f"❌ Transcription failed: {e}")
                step += 1
                progress_bar.progress(step / total_steps, "Transcription Complete")
                console_placeholder.markdown(f'<div class="console-output">{"<br>".join(analyzer.console_messages)}</div>', unsafe_allow_html=True)

            if do_analyze_eye_contact:
                status_text.text("👁️ Analyzing eye contact...")
                try:
                    analyzer.log_to_console("Running eye contact analysis...")
                    results['eye_contact'] = analyze_eye_contact(video_path)
                    analyzer.log_to_console("✅ Eye contact analysis complete.")
                except Exception as e:
                    analyzer.log_to_console(f"❌ Eye contact analysis failed: {e}")
                step += 1
                progress_bar.progress(step / total_steps, "Eye Contact Analyzed")
                console_placeholder.markdown(f'<div class="console-output">{"<br>".join(analyzer.console_messages)}</div>', unsafe_allow_html=True)

            if generate_final_report:
                status_text.text("📊 Generating final report...")
                try:
                    analyzer.log_to_console("Running final LLM analysis...")
                    emotion_summary = analyzer.summarize_emotion_data(results.get('emotions', []))
                    results['final_analysis'] = analyze_student_pitch(
                        emotion_summary, results.get('audio'), results.get('transcription'), results.get('eye_contact')
                    )
                    analyzer.log_to_console("✅ Final analysis complete.")
                except Exception as e:
                    analyzer.log_to_console(f"❌ Final analysis failed: {e}")
                step += 1
                progress_bar.progress(step / total_steps, "Final Report Generated")
                console_placeholder.markdown(f'<div class="console-output">{"<br>".join(analyzer.console_messages)}</div>', unsafe_allow_html=True)


            progress_bar.progress(1.0, "Analysis Complete!")
            status_text.success("✅ Analysis Complete! View the results below.")
            analyzer.log_to_console("🎉 All selected analyses have finished.")
            console_placeholder.markdown(f'<div class="console-output">{"<br>".join(analyzer.console_messages)}</div>', unsafe_allow_html=True)
            
            st.session_state.analysis_results = results
            st.session_state.analysis_complete = True
            st.session_state.start_analysis = False
            st.rerun()

        except Exception as e:
            st.error(f"A critical error occurred during the analysis pipeline: {e}")
            analyzer.log_to_console(f"💥 CRITICAL PIPELINE ERROR: {e}")
        finally:
            if video_path and os.path.exists(video_path):
                os.unlink(video_path)
            if audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)
    
    if st.session_state.get('analysis_complete'):
        st.markdown("---")
        st.markdown("# 📊 **ANALYSIS RESULTS**")
        results = st.session_state.analysis_results
        
        tab_names = [name for name, res_key in {
            "🎭 Emotions": "emotions", "🎤 Audio": "audio",
            "📝 Transcription": "transcription", "👁️ Eye Contact": "eye_contact",
            "📋 Full Report": "report"  # A placeholder key
        }.items() if res_key in results or res_key == 'report']
        
        if not results:
            st.warning("No analysis was performed or all analyses failed. Please check the console log.")
            return

        tabs = st.tabs(tab_names)
        tab_map = {name: t for name, t in zip(tab_names, tabs)}

        if '🎭 Emotions' in tab_map and 'emotions' in results:
            with tab_map['🎭 Emotions']:
                st.markdown("## Emotion Analysis Results")
                fig_pie, fig_timeline, df = analyzer.create_emotion_visualization(results['emotions'])
                if fig_pie:
                    col1, col2 = st.columns(2)
                    col1.plotly_chart(fig_pie, use_container_width=True)
                    col2.plotly_chart(fig_timeline, use_container_width=True)
                    with st.expander("Show Raw Emotion Data"):
                        st.dataframe(df)
                else:
                    st.warning("No emotion data was generated.")
        
        if '🎤 Audio' in tab_map and 'audio' in results:
            with tab_map['🎤 Audio']:
                st.markdown("## Audio Analysis Results")
                fig_audio = analyzer.create_audio_visualization(results['audio'])
                if fig_audio:
                    st.plotly_chart(fig_audio, use_container_width=True)
                else:
                    st.warning("Could not visualize audio data.")
                
                with st.expander("Show Raw Audio Data (JSON)"):
                    st.json(results['audio'])

        if '📝 Transcription' in tab_map and 'transcription' in results:
            with tab_map['📝 Transcription']:
                st.markdown("## Transcription Results")
                transcription = results.get('transcription', "Not available.")
                st.markdown(f'<div class="analysis-card"><p>{transcription}</p></div>', unsafe_allow_html=True)
                words = transcription.split()
                if len(words) > 10:
                    word_freq = Counter(w.lower().strip('.,!?') for w in words if len(w) > 3)
                    if word_freq:
                        freq_df = pd.DataFrame(word_freq.most_common(15), columns=['Word', 'Frequency'])
                        fig_words = px.bar(freq_df, x='Frequency', y='Word', orientation='h', title="Most Frequent Words")
                        fig_words.update_layout(paper_bgcolor='rgba(0,0,0,0)', yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig_words, use_container_width=True)

        if '👁️ Eye Contact' in tab_map and 'eye_contact' in results:
             with tab_map['👁️ Eye Contact']:
                 st.markdown("## Eye Contact Analysis Results")
                 st.markdown(f"<div class='analysis-card'>{results['eye_contact']}</div>", unsafe_allow_html=True)

        if '📋 Full Report' in tab_map:
            with tab_map['📋 Full Report']:
                st.markdown("## Comprehensive Performance Report")
                comprehensive_report = analyzer.create_comprehensive_analysis_report(results)
                st.markdown(comprehensive_report, unsafe_allow_html=True)
                
                pdf_buffer = analyzer.create_pdf_report(comprehensive_report)
                st.download_button(
                    label="📥 Download Full Report as PDF",
                    data=pdf_buffer,
                    file_name=f"pitch_analysis_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; font-family: 'Rajdhani', sans-serif;">
        <p>🚀 Powered by Advanced AI • Built with Streamlit • © 2024 Pitch Performance Analyzer</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()