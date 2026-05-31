from TTS.api import TTS
import os
import requests
import chromadb
from chromadb.utils import embedding_functions

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

    elif cmd == "export chat":
        docs = conversation_collection.get()["documents"]
        with open("chat_export.txt", "w", encoding="utf-8") as f:
            for doc in docs:
                f.write(doc + "\n")
        return "Chat exported to chat_export.txt."

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
        # Retrieve ALL stored documents from ChromaDB
        all_docs = conversation_collection.get()["documents"]

        # Limit to last 5 entries
        last_docs = all_docs[-5:] if all_docs else []
        previous_context = "\n".join(last_docs)

        # Build messages for OpenRouter using current personality
        messages = [
            {"role": "system", "content": personalities[current_personality]}
        ]

        # Inject memory as assistant role (so AI sees it as its own past responses)
        if previous_context:
            messages.append({"role": "assistant", "content": f"{previous_context}"})

        # Add current user prompt
        messages.append({"role": "user", "content": user_prompt})

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

        # --- EMOJI-SAFE TTS FIX ---
        safe_text = ''.join(c for c in ai_text if c.isprintable() and ord(c) < 128)

        # Convert AI response to speech and save
        tts.tts_to_file(text=safe_text, file_path="response_audio.wav")
        print("✅ Voice output saved as response_audio.wav")

        # Play audio automatically
        os.startfile("response_audio.wav")

    except Exception as e:
        print("❌ API error:", e)
