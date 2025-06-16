import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from the .env file
load_dotenv()

def analyze_student_pitch(emotion_data, audio_data, transcribed_text,eye_contact_data):
    """
    Analyze student pitch performance and return AI-generated summary
    
    Args:
        emotion_data (str): Facial emotion analysis results
        audio_data (dict): Audio feature analysis results
        transcribed_text (str): Speech-to-text transcription
    
    Returns:
        str: AI-generated analysis summary
    """
    
    # Initialize Groq client
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    except Exception as e:
        return f"Error initializing Groq client: {e}"
    
    # Create analysis prompt
    prompt_template = f"""
You are a presentation coach analyzing a student's pitch. Based on the data below, provide a CONCISE SUMMARY (maximum 300 words) with key insights and top recommendations.

**DATA:**
Emotion Analysis: {emotion_data}
Audio Analysis: {audio_data}
Speech Content: {transcribed_text}
Eye Contact Analysis: {eye_contact_data}

**REQUIRED SUMMARY FORMAT:**

**PERFORMANCE SCORE:** [X/10]

**KEY INSIGHTS:**
- Confidence Level: [Brief assessment]
- Vocal Delivery: [Key observation]
- Content Quality: [Main strength/weakness]

**TOP 3 IMPROVEMENTS:**
1. [Most important fix]
2. [Second priority]
3. [Third priority]

**STRENGTHS TO MAINTAIN:**
- [1-2 positive aspects]

**NEXT STEPS:**
[1-2 specific actions for immediate improvement]

Keep the summary concise, actionable, and encouraging.
"""
    
    try:
        # Create completion
        completion = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[
                {
                    "role": "system",
                    "content": "You are a presentation coach. Provide concise, actionable feedback summaries in exactly the requested format. Keep responses under 300 words total."
                },
                {
                    "role": "user",
                    "content": prompt_template
                }
            ],
            temperature=0.5,
            max_tokens=4096,
            top_p=0.9,
            stream=False,  # Changed to False to return complete response
            stop=None,
        )
        
        # Return the complete analysis
        return completion.choices[0].message.content
        
    except Exception as e:
        return f"Error during analysis: {e}"

# Example usage when called from another script:
# from pitch_analyzer import analyze_student_pitch
# result = analyze_student_pitch(emotion_summary, audio_result, audio_text)
# print(result)