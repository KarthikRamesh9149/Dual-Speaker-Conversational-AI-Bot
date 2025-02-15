# Import necessary libraries
import os
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import speech_recognition as sr
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import requests

# Load environment variables from .env
load_dotenv()

# API keys
GROQCLOUD_API_KEY = os.getenv("GROQCLOUD_API_KEY", st.secrets.get("GROQCLOUD_API_KEY"))
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", st.secrets.get("TAVILY_API_KEY"))

# Initialize LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key=GROQCLOUD_API_KEY,
    model_name="llama-3.1-70b-versatile"
)

# Initialize Embeddings
embeddings = HuggingFaceEmbeddings()

# Initialize Session State
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Persistent Directory for ChromaDB
CHROMA_DB_DIR = "./chroma_db"

# Speech Recognizer
recognizer = sr.Recognizer()

# Helper Functions
def transcribe_audio(audio_bytes):
    """Transcribes audio bytes using SpeechRecognition."""
    try:
        temp_audio_path = "temp_audio.wav"
        with open(temp_audio_path, "wb") as f:
            f.write(audio_bytes)
        with sr.AudioFile(temp_audio_path) as source:
            audio = recognizer.record(source)
            transcription = recognizer.recognize_google(audio)
            return transcription
    except sr.UnknownValueError:
        return "Could not understand the audio."
    except sr.RequestError:
        return "Speech recognition service unavailable."

def identify_question(text):
    """Identify embedded questions in text."""
    prompt = f"Identify the question embedded in the following statement:\n{text}"
    response = llm.invoke(prompt)
    return response.content.strip()

def decide_action(question):
    """Decide whether to use Web Search or LLM."""
    decision_prompt = f"""
    You are tasked with deciding the action for answering the following question:
    Question: {question}
    Decide whether to use 'Web Search' or 'LLM' to answer.
    Provide one of these answers: 'Web Search', 'LLM'.
    """
    response = llm.invoke(decision_prompt)
    return response.content.strip()

def web_search(question):
    """Perform web search using Tavily API."""
    response = requests.post(
        'https://api.tavily.com/search',
        json={"query": question, "num_results": 3},
        headers={"Authorization": f"Bearer {TAVILY_API_KEY}"}
    )
    if response.status_code == 200:
        results = response.json().get("results", [])
        summaries = [result['snippet'] for result in results]
        return " ".join(summaries[:3])[:100]  # Limit to 100 words
    return "Unable to perform web search at the moment."

def trim_answer(answer, word_limit=100):
    """Trim answer to a specified word limit."""
    words = answer.split()
    return " ".join(words[:word_limit])

# Streamlit UI
st.title("AI Assistant with Agentic RAG Framework")
st.write("Supports RAG, Web Search, and direct LLM responses.")

# PDF Upload for RAG
uploaded_pdf = st.file_uploader("Upload a PDF for RAG (optional)", type=["pdf"])
if uploaded_pdf and not st.session_state.pdf_uploaded:
    with open("uploaded_document.pdf", "wb") as f:
        f.write(uploaded_pdf.read())
    st.success("PDF uploaded successfully! Processing...")
    loader = PyPDFLoader("uploaded_document.pdf")
    documents = loader.load()

    # Initialize Chroma vector store with a persistent directory
    if not os.path.exists(CHROMA_DB_DIR):
        os.makedirs(CHROMA_DB_DIR)

    st.session_state.vector_store = Chroma.from_documents(
        documents,
        embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    st.session_state.vector_store.persist()
    st.session_state.pdf_uploaded = True

vector_store = st.session_state.vector_store

# Conversation Input Loop
st.subheader("🎤 Speak Now")
webrtc_ctx = webrtc_streamer(key="audio", mode="sendonly")
if webrtc_ctx.audio_receiver:
    audio_frames = webrtc_ctx.audio_receiver.get_frames()
    for frame in audio_frames:
        audio_data = frame.to_ndarray()
        transcription = transcribe_audio(audio_data)
        if transcription:
            st.session_state.conversation.append(("Speaker", transcription))

# Display Conversation
if st.session_state.conversation:
    st.subheader("**Conversation History:**")
    for speaker, line in st.session_state.conversation:
        st.write(f"{speaker}: {line}")

# Processing and Answer Generation
if st.button("Help Me!"):
    st.subheader("Generating Answer...")

    full_conversation = " ".join([line for _, line in st.session_state.conversation])
    st.write("**Combined Transcription:**")
    st.code(full_conversation)

    question = identify_question(full_conversation)
    st.write("**Question Identified:**")
    st.info(question)

    final_source = None  # Track the source of the answer

    # Check RAG
    if vector_store:
        st.write("Checking PDF context (RAG)...")
        retriever = vector_store.as_retriever()
        qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever)
        answer = qa_chain.run({"question": question, "chat_history": []})
        if answer and "I don't know" not in answer:
            final_source = "RAG"

    # If RAG fails, decide between Web Search and LLM
    if not final_source:
        decision = decide_action(question)
        st.write("**LLM Decision:**")
        st.warning(decision)

        if decision == "Web Search":
            answer = web_search(question)
            final_source = "Web Search"
        else:
            response = llm.invoke(question)
            answer = response.content
            final_source = "LLM"

    # Display the final answer and source
    trimmed_answer = trim_answer(answer, word_limit=100)
    st.subheader("**Answer:**")
    st.success(trimmed_answer)

    st.write(f"**Source of Answer:** {final_source}")

# Clear History
if st.button("Clear Conversation History"):
    st.session_state.conversation = []
    st.session_state.pdf_uploaded = False
    st.session_state.vector_store = None
    st.success("Conversation history cleared.")
