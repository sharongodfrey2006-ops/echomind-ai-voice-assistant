from TTS.api import TTS
import os
import requests
import chromadb
from chromadb.utils import embedding_functions
from chat_db import save_chat, export_txt, export_json, export_md   # database functions
import whisper
import hashlib

# --- PERFORMANCE OPTIMIZATION SETUP ---

# Load Whisper model once (faster transcription)
def load_whisper_model():
    return whisper.load_model("base")   # smaller model for speed

whisper_model = load_whisper_model()

# Audio cache dictionary
audio_cache = {}

def get_audio_hash(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return hashlib.md5(data).hexdigest()

def transcribe_audio(file_path):
    """Transcribe audio with caching to avoid re-processing same file."""
    file_hash = get_audio_hash(file_path)
    if file_hash in audio_cache:
        return audio_cache[file_hash]
    result = whisper_model.transcribe(file_path)
    audio_cache[file_hash] = result["text"]
    return result["text"]

# --- STEP 28: TOKEN MANAGEMENT ---
def count_tokens(messages):
    """Approximate token count based on word splits."""
    return sum(len(m["content"].split()) for m in messages)

def trim_conversation(messages, max_tokens=3000):
    """Trim oldest messages until under max_tokens."""
    while count_tokens(messages) > max_tokens and len(messages) > 2:
        # remove oldest user/assistant message (keep system prompt)
        messages.pop(1)
    return messages

# Load OpenRouter API key from environment variables
openrouter_key = os.getenv("OPENROUTER_API_KEY")

# Initialize TTS model
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

# Initialize ChromaDB client with HuggingFace embeddings
client = chromadb.Client()
conversation_collection = client.get_or_create_collection(
    name="conversation_memory",
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
)

# Conversation history list
conversation = []

# --- RESET FUNCTION ---
def reset_memory():
    """Clear all stored memory in ChromaDB."""
    ids = conversation_collection.get()["ids"]
    if ids:
        conversation_collection.delete(ids=ids)
        print("🧹 Memory reset: All stored documents cleared.")
    else:
        print("🧹 Memory reset: No documents to clear.")

# --- STEP 19: PERSONALITY DEFINITIONS ---
personalities = {
    "professional": "You are a professional assistant. Respond formally, concisely, and with clear structure.",
    "friendly": "You are a friendly assistant. Respond warmly, casually, and with lots of encouragement.",
    "technical": "You are a technical expert. Respond with detailed explanations, code snippets, and precise terminology.",
    "coach": "You are a productivity coach. Respond with motivational tone, step-by-step guidance, and actionable advice."
}

# --- STARTUP PERSONALITY SELECTOR ---
print("Available personalities: professional, friendly, technical, coach")
choice = input("Select a personality to start with: ").strip().lower()
if choice in personalities:
    current_personality = choice
else:
    print("⚠️ Personality not recognized. Defaulting to 'friendly'.")
    current_personality = "friendly"

print(f"✨ Starting with personality: {current_personality.capitalize()}")

# --- VOICE COMMAND HANDLER ---
def handle_command(command):
    """Detect and execute special voice/text commands."""
    cmd = command.lower().strip()

    if cmd == "open chat history":
        docs = conversation_collection.get()["documents"]
        print("📖 Chat History:")
        for i, doc in enumerate(docs, 1):
            print(f"{i}. {doc}")
        return "Chat history opened."

    elif cmd == "clear conversation":
        reset_memory()
        return "Conversation cleared."

    elif cmd.startswith("change personality"):
        parts = cmd.split()
        if len(parts) >= 3:
            choice = parts[2]
            if choice in personalities:
                global current_personality
                current_personality = choice
                return f"Personality changed to {choice.capitalize()}."
            else:
                return "⚠️ Personality not recognized."
        return "⚠️ Usage: change personality <professional|friendly|technical|coach>"

    elif cmd == "generate summary":
        docs = conversation_collection.get()["documents"]
        summary = " | ".join(docs[-5:]) if docs else "No conversation yet."
        print("📝 Summary:", summary)
        return "Summary generated."

    elif cmd == "export txt":
        export_txt()
        return "Chat exported to chat_export.txt."

    elif cmd == "export json":
        export_json()
        return "Chat exported to chat_export.json."

    elif cmd == "export md":
        export_md()
        return "Chat exported to chat_export.md."

    return None

# --- MAIN LOOP ---
while True:
    # Ask user for input interactively
    user_prompt = input("\nYou: ")

    # Exit option
    if user_prompt.lower() in ["exit", "quit"]:
        print("👋 Ending chat. Memory remains stored in ChromaDB.")
        break

    # Reset option
    if user_prompt.lower() == "reset memory":
        reset_memory()
        continue

    # --- STEP 21 & 22: COMMAND DETECTION ---
    command_response = handle_command(user_prompt)
    if command_response:
        print("🔊 Voice Command:", command_response)
        # Voice confirmation
        safe_text = ''.join(c for c in command_response if c.isprintable() and ord(c) < 128)
        tts.tts_to_file(text=safe_text, file_path="response_audio.wav")
        os.startfile("response_audio.wav")
        continue

    # Personality switch option (typed)
    if user_prompt.lower().startswith("set personality"):
        parts = user_prompt.split()
        if len(parts) >= 3:
            choice = parts[2].lower()
            if choice in personalities:
                current_personality = choice
                print(f"✨ Personality switched to: {choice.capitalize()}")
            else:
                print("⚠️ Personality not recognized. Options: professional, friendly, technical, coach")
        else:
            print("⚠️ Usage: set personality <professional|friendly|technical|coach>")
        continue

    try:
        # --- PERFORMANCE OPTIMIZATION: Efficient memory retrieval ---
        results = conversation_collection.query(
            query_texts=[user_prompt],
            n_results=5
        )

        # Flatten list of lists into plain text
        docs = results.get("documents", [])
        flat_docs = [doc for sublist in docs for doc in sublist]
        relevant_context = "\n".join(flat_docs) if flat_docs else ""

        # Build messages for OpenRouter using current personality
        messages = [
            {"role": "system", "content": personalities[current_personality]}
        ]

        # Inject relevant memory
        if relevant_context:
            messages.append({"role": "assistant", "content": relevant_context})

        # Add current user prompt
        messages.append({"role": "user", "content": user_prompt})

        # --- STEP 28: Apply token management ---
        messages = trim_conversation(messages)

        # Debug: show token count for proof
        print("🔢 Token count:", count_tokens(messages))

        # Send to OpenRouter
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {openrouter_key}"},
            json={"model": "openai/gpt-3.5-turbo", "messages": messages}
        )

        # Extract AI response
        ai_text = response.json()["choices"][0]["message"]["content"]
        print(f"{current_personality.capitalize()} AI:", ai_text)

        # Add AI response to conversation
        conversation.append({"role": "assistant", "content": ai_text})

        # Store both user and AI response in ChromaDB
        conversation_collection.add(documents=[user_prompt], ids=[f"user_{len(conversation)}"])
        conversation_collection.add(documents=[ai_text], ids=[f"ai_{len(conversation)}"])

        # 🔎 Inspect stored memory
        print("🔎 Stored documents:", conversation_collection.get())

        # --- Save to SQLite database ---
        save_chat(user_prompt, ai_text, current_personality)

        # --- EMOJI-SAFE TTS FIX ---
        safe_text = ''.join(c for c in ai_text if c.isprintable() and ord(c) < 128)

        # Convert AI response to speech and save
        tts.tts_to_file(text=safe_text, file_path="response_audio.wav")
        print("✅ Voice output saved as response_audio.wav")

        # Play audio automatically
        os.startfile("response_audio.wav")

    except Exception as e:
        print("❌ API error:", e)
