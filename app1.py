import os
import whisper

from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain.chains import RetrievalQA


# -----------------------------------
# STEP 1: Transcribe Video
# -----------------------------------
def transcribe_video(video_path):

    print("Loading Whisper model...")
    model = whisper.load_model("base")

    print("Transcribing video...")
    result = model.transcribe(video_path)

    return result["text"]


# -----------------------------------
# STEP 2: Create Vector Database
# -----------------------------------
def create_vector_db(transcript):

    print("Splitting transcript into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.create_documents([transcript])

    print("Creating embeddings...")

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    print("Storing in ChromaDB...")

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./video_db"
    )

    return vectorstore


# -----------------------------------
# STEP 3: Load LLM + Retrieval Chain
# -----------------------------------
def create_qa_chain(vectorstore):

    retriever = vectorstore.as_retriever()

    llm = ChatOllama(
        model="llama3"
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    return qa_chain


# -----------------------------------
# STEP 4: MAIN
# -----------------------------------
video_path = r"D:\youtube-summarizer\input_video.mp4"

if not os.path.exists(video_path):
    print("❌ Video file not found.")
    exit()


# -----------------------------------
# TRANSCRIBE
# -----------------------------------
print("\n========== STEP 1 ==========")
transcript = transcribe_video(video_path)

print("\n✅ Transcript generated")



# -----------------------------------
# CREATE VECTOR DATABASE
# -----------------------------------
print("\n========== STEP 2 ==========")

vectorstore = create_vector_db(transcript)

print("✅ Vector database created")


# -----------------------------------
# LOAD QA SYSTEM
# -----------------------------------
print("\n========== STEP 3 ==========")

qa_chain = create_qa_chain(vectorstore)

print("✅ Video Chat Ready!")


# -----------------------------------
# CHAT LOOP
# -----------------------------------
print("\n========== CHAT WITH VIDEO ==========")
print("Type 'exit' to quit.\n")

while True:

    question = input("You: ")

    if question.lower() == "exit":
        break

    print("\nThinking...\n")

    response = qa_chain.invoke({
        "query": question
    })

    print("Assistant:")
    print(response["result"])
    print()