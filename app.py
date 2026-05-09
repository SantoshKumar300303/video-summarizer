import os
import whisper
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

# -----------------------------
# STEP 1: Transcribe video (NO ffmpeg command needed)
# -----------------------------
def transcribe_video(video_path):
    print("Loading Whisper model...")
    model = whisper.load_model("base")  # tiny / base / small / medium

    print("Transcribing video...")
    result = model.transcribe(video_path)

    return result["text"]


# -----------------------------
# STEP 2: Prompt template
# -----------------------------
prompt = PromptTemplate.from_template("""
You are a helpful assistant.

Summarize the following video transcript:

{transcript}

Provide:
1. Short Summary
2. Important Points
3. Final Conclusion
""")


# -----------------------------
# STEP 3: LLM (Ollama)
# -----------------------------
llm = ChatOllama(model="llama3")


# -----------------------------
# STEP 4: MAIN
# -----------------------------
video_path = r"D:\youtube-summarizer\input_video.mp4"   # 👈 put your file here

if not os.path.exists(video_path):
    print("❌ Video file not found. Put input_video.mp4 in the same folder.")
    exit()

# Transcribe
transcript = transcribe_video(video_path)

print("\n✅ Transcript generated\n")

# Summarize
final_prompt = prompt.format(transcript=transcript)

print("Generating summary...\n")
response = llm.invoke(final_prompt)

print("\n========== SUMMARY ==========\n")
print(response.content)