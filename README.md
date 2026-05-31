# EchoMind AI Voice Assistant

## Project Overview
EchoMind AI Voice Assistant is a conversational AI system that integrates speech-to-text, large language models, and text-to-speech to create natural voice interactions. It allows users to speak directly to the assistant, receive intelligent responses, and hear those responses spoken back.

## Features
- 🎤 Voice input recording with SoundDevice
- 📝 Speech-to-text transcription using Whisper
- 🤖 AI conversation pipeline powered by OpenRouter LLMs
- 🔊 Text-to-speech output with Coqui TTS
- 🧠 Memory system using ChromaDB
- 🎭 Personality switching (Professional, Friendly, Technical, Productivity Coach)
- 📜 Chat history storage and export (TXT, JSON, Markdown)
- 🖥️ Streamlit dashboard frontend

## AI Architecture
**Pipeline:**  
Voice Input → Whisper STT → LLM Response → TTS → Voice Output  

Supporting modules:  
- Memory (ChromaDB) for context continuity  
- Personality system for dynamic tone switching  
- Command detection for voice‑based actions  

## Voice Processing Pipeline
1. Record audio with SoundDevice  
2. Process audio with PyDub (format conversion, normalization, noise reduction)  
3. Transcribe speech with Whisper  
4. Generate AI response via OpenRouter API  
5. Convert text to voice with Coqui TTS  
6. Play audio output and save recordings  

## Installation Steps
```bash
# Clone repository
git clone https://github.com/sharongodfrey2006-ops/echomind-ai-voice-assistant
cd echomind-ai-voice-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
