import os
import whisper
from groq import Groq
from gtts import gTTS
import streamlit as st
from st_audiorec import st_audiorec
import tempfile

# Set your Groq API key here
os.environ['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'] = 'your_actual_groq_api_key'  # Replace with your API key

# Load Whisper model for transcription
model = whisper.load_model("base")

# Initialize Groq API client
client = Groq(api_key=os.environ.get("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"))

# Function to transcribe audio using Whisper
def transcribe_audio(audio_file):
    result = model.transcribe(audio_file)
    return result['text']

# Function to interact with LLM using Groq API
def get_llm_response(input_text):
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": input_text}],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

# Function to convert LLM response text to speech using gTTS
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        audio_file = tmp.name + '.mp3'
        tts.save(audio_file)
    return audio_file

# Streamlit UI
st.title("Real-time Voice-to-Voice Chatbot")
st.write("This chatbot transcribes your speech, processes it with an LLM, and returns the response as both text and speech.")

# Audio recording section
st.subheader("Record Your Audio")
audio_data = st_audiorec.record_file("audio_recording.wav")

if audio_data is not None:
    # Save uploaded audio file to a temporary location
    with open("audio_recording.wav", "wb") as f:
        f.write(audio_data)

    # Step 1: Transcribe audio to text
    transcribed_text = transcribe_audio("audio_recording.wav")
    st.write("Transcribed Text:", transcribed_text)

    # Step 2: Get LLM response from Groq API
    llm_response = get_llm_response(transcribed_text)
    st.write("LLM Response:", llm_response)

    # Step 3: Convert LLM response to audio
    audio_output_path = text_to_speech(llm_response)
    
    # Step 4: Provide audio output
    st.audio(audio_output_path)
