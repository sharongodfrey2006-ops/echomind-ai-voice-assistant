# app.py
import streamlit as st
import os
import requests
from TTS.api import TTS
import chromadb
from chromadb.utils import embedding_functions
from chat_db import export_txt, export_json, export_md   # NEW: import export functions

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Voice Assistant", layout="wide")

# --- INITIAL SETUP ---
openrouter_key = os.getenv("OPENROUTER_API_KEY")
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
client = chromadb.Client()
conversation_collection = client.get_or_create_collection(
    name="conversation_memory",
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
)

# --- PERSONALITIES ---
personalities = {
    "professional": "You are a professional assistant. Respond formally, concisely, and with clear structure.",
    "friendly": "You are a friendly assistant. Respond warmly, casually, and with lots of encouragement.",
    "technical": "You are a technical expert. Respond with detailed explanations, code snippets, and precise terminology.",
    "coach": "You are a productivity coach. Respond with motivational tone, step-by-step guidance, and actionable advice."
}

# --- UI ELEMENTS ---
st.title("🎙️ AI Voice Assistant Dashboard")
st.sidebar.header("Settings")

# Personality selector
current_personality = st.sidebar.selectbox("Select Personality", list(personalities.keys()), index=1)

# Voice recording button (placeholder)
st.sidebar.button("🎤 Record Voice")

# Chat history display
st.subheader("💬 Chat History")
docs = conversation_collection.get()["documents"]
if docs:
    for doc in docs[-10:]:
        st.text(doc)
else:
    st.info("No conversation yet.")

# Input area
user_prompt = st.text_input("Type your message:")

# --- RESPONSE PANEL ---
if st.button("Send"):
    if user_prompt:
        # Retrieve memory
        all_docs = conversation_collection.get()["documents"]
        last_docs = all_docs[-5:] if all_docs else []
        previous_context = "\n".join(last_docs)

        messages = [
            {"role": "system", "content": personalities[current_personality]}
        ]
        if previous_context:
            messages.append({"role": "assistant", "content": f"{previous_context}"})
        messages.append({"role": "user", "content": user_prompt})

        with st.spinner("Generating response..."):
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {openrouter_key}"},
                json={"model": "openai/gpt-3.5-turbo", "messages": messages}
            )
            ai_text = response.json()["choices"][0]["message"]["content"]

        st.success(ai_text)

        # Store conversation
        conversation_collection.add(documents=[user_prompt], ids=[f"user_{len(all_docs)+1}"])
        conversation_collection.add(documents=[ai_text], ids=[f"ai_{len(all_docs)+1}"])

        # Audio playback
        safe_text = ''.join(c for c in ai_text if c.isprintable() and ord(c) < 128)
        tts.tts_to_file(text=safe_text, file_path="response_audio.wav")
        st.audio("response_audio.wav")

    else:
        st.warning("Please enter a message before sending.")

# --- EXPORT SECTION ---
st.subheader("📤 Export Chat History")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Export TXT"):
        export_txt()
        with open("chat_export.txt", "r", encoding="utf-8") as f:
            st.download_button("⬇️ Download TXT", f, file_name="chat_export.txt")

with col2:
    if st.button("Export JSON"):
        export_json()
        with open("chat_export.json", "r", encoding="utf-8") as f:
            st.download_button("⬇️ Download JSON", f, file_name="chat_export.json")

with col3:
    if st.button("Export Markdown"):
        export_md()
        with open("chat_export.md", "r", encoding="utf-8") as f:
            st.download_button("⬇️ Download Markdown", f, file_name="chat_export.md")
