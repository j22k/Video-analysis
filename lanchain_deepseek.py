import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from the .env file
# This line looks for a .env file in the current directory and loads its content
# into the environment, making os.getenv("GROQ_API_KEY") work.
load_dotenv()

def analyze_student_pitch(emotion_data, audio_data, transcribed_text):
    """
    Analyze student pitch performance based on emotion detection, audio analysis, and speech content
    """
    
    # Create prompt template for comprehensive pitch analysis
    prompt_template = f"""
You are an expert communication coach analyzing a student's pitch presentation. Based on the following data from video, audio, and speech content analysis, provide a comprehensive assessment and improvement recommendations.

**EMOTION ANALYSIS DATA:**
{emotion_data}

**AUDIO ANALYSIS DATA:**
{audio_data}

**TRANSCRIBED SPEECH CONTENT:**
{transcribed_text}

Please provide a detailed analysis covering:

1. **OVERALL PERFORMANCE ASSESSMENT:**
   - Current confidence level interpretation
   - Emotional engagement analysis
   - Voice quality evaluation

2. **DETAILED DATA INTERPRETATION:**
   - What the emotion confidence scores reveal about the student's state
   - Audio metrics analysis (pitch, speaking rate, volume, pauses, voice quality)
   - Speech content analysis (clarity, structure, persuasiveness, filler words)
   - Correlation between emotional state, vocal delivery, and message content

3. **AREAS FOR IMPROVEMENT:**
   - Confidence building strategies
   - Voice modulation techniques
   - Pacing and pause optimization
   - Emotional engagement enhancement
   - Content structure and clarity
   - Persuasive language techniques
   - Reduction of filler words and hesitations

4. **SPECIFIC RECOMMENDATIONS:**
   - Practice exercises for voice improvement
   - Techniques to increase emotional confidence
   - Strategies for better audience engagement
   - Methods to reduce nervousness indicators
   - Content restructuring suggestions
   - Language and vocabulary enhancement
   - Storytelling and persuasion techniques

5. **ACTION PLAN:**
   - Short-term improvements (next presentation)
   - Long-term development goals
   - Measurable targets for improvement

6. **STRENGTHS TO MAINTAIN:**
   - Positive aspects identified from the data
   - What the student is doing well
   - Strong content elements to build upon
   - Effective communication patterns to reinforce

7. **PITCH CONTENT EVALUATION:**
   - Message clarity and coherence
   - Persuasive elements present
   - Target audience alignment
   - Call-to-action effectiveness
   - Supporting evidence and examples
   - Overall pitch structure assessment

Format your response in a clear, actionable manner that a student can easily understand and implement.
"""
    
    return prompt_template

# Initialize Groq client
# The Groq() client will automatically look for the 'GROQ_API_KEY' environment variable.
# Since load_dotenv() set it, this will work without passing the key explicitly.
client = Groq()

# For even more explicit code, you can do this:
# client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# Your input data
emotion_input = """Frame 1263: Emotion = Happiness (Confidence: 32.00%)
Frame 1264: Emotion = Happiness (Confidence: 32.00%)
Frame 1265: Emotion = Happiness (Confidence: 32.00%)
Frame 1266: Emotion = Happiness (Confidence: 32.00%)
Frame 1267: Emotion = Happiness (Confidence: 32.00%)
Frame 1268: Emotion = Happiness (Confidence: 32.00%)
Frame 1269: Emotion = Happiness (Confidence: 32.00%)
Frame 1270: Emotion = Happiness (Confidence: 32.00%)
Frame 1271: Emotion = Happiness (Confidence: 32.00%)
Frame 1272: Emotion = Happiness (Confidence: 32.00%)"""

audio_input = """--- Audio Analysis Results ---
pitch_variation: {'mean_pitch': 220.41992509093834, 'pitch_std': 103.94065546928185, 'pitch_range': 519.7445832166878}
speaking_rate: 1.21
volume_consistency: {'rms_mean': 0.02083481289446354, 'rms_std': 0.017533134669065475}
pause_analysis: {'num_pauses': 20, 'avg_pause_duration': 0.33901133786848076}
voice_quality: {'jitter': 0.03206, 'shimmer': 0.17567}"""

# Add transcribed text variable (replace with your actual transcribed content)
transcribed_text = """[REPLACE WITH YOUR ACTUAL TRANSCRIBED TEXT FROM THE STUDENT'S PITCH]

Example format:
"Hello everyone, um, today I want to, uh, present my idea about... well, it's about creating a mobile app that helps students, you know, manage their time better. So, the problem is that many students struggle with, um, organizing their schedules and, uh, they often miss deadlines..."
"""

# Generate the analysis prompt
analysis_prompt = analyze_student_pitch(emotion_input, audio_input, transcribed_text)

# Create completion with the analysis prompt
completion = client.chat.completions.create(
    # Note: I've updated the model to a more recent and powerful one available on Groq.
    # You can change it back to "deepseek-r1-distill-llama-70b" if you prefer.
    model="llama3-70b-8192", 
    messages=[
        {
            "role": "user",
            "content": analysis_prompt
        }
    ],
    temperature=0.6,
    max_tokens=4096, # Renamed from max_completion_tokens
    top_p=0.95,
    stream=True,
    stop=None,
)

print("=== STUDENT PITCH PERFORMANCE ANALYSIS ===\n")
for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")

print("\n\n=== END OF ANALYSIS ===")