# Video Analysis Tool 🎥

An AI-powered video interview analysis system that provides comprehensive feedback on presentation skills, emotion detection, eye contact, and speech patterns.

## 🚀 Features

- **Emotion Detection**: Real-time facial emotion recognition using deep learning
- **Eye Contact Tracking**: Monitors and analyzes eye contact patterns throughout the video
- **Audio Analysis**: Transcribes speech and analyzes audio quality
- **Filler Word Detection**: Identifies and counts filler words (um, uh, like, etc.)
- **Interactive UI**: Built with Streamlit and Tkinter for easy use
- **AI-Powered Insights**: Integration with DeepSeek LangChain for intelligent analysis

## 📋 Requirements

```bash
pip install -r requirements.txt
```
Key Dependencies
OpenCV for video processing
TensorFlow/PyTorch for emotion detection
Streamlit for web interface
Speech recognition libraries
MediaPipe for facial landmarks
🎯 Usage
Streamlit Interface
bash
streamlit run StramlitUI.py
Tkinter Application
bash
python tkinter_app.py
Command Line
bash
python main.py --video path/to/video.mp4
📁 Project Structure
Code
Video-analysis/
├── StramlitUI.py              # Streamlit web interface
├── tkinter_app.py             # Desktop GUI application
├── main.py                    # Main processing script
├── emotion_detection/         # Emotion recognition module
├── eye_contact.py             # Eye contact tracking
├── eye_contact_optimized.py   # Optimized version
├── audio_analysis.py          # Audio processing
├── filler_word_summary.py     # Filler word detection
├── transcribe_audio.py        # Speech-to-text
├── lanchain_deepseek.py       # AI integration
└── requirements.txt           # Dependencies
🔧 Configuration
Create a .env file for API keys:

Code
DEEPSEEK_API_KEY=your_api_key_here
📊 Output
The tool provides:

Emotion distribution charts
Eye contact percentage and timeline
Filler word count and frequency
Audio transcription
Overall presentation score
AI-generated improvement suggestions
🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

📝 License
This project is open source and available under the MIT License.
