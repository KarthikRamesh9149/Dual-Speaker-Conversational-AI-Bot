Dual-Speaker Conversational AI Bot

The Dual-Speaker Conversational AI Bot is an advanced Agentic RAG (Retrieval-Augmented Generation) AI Assistant designed to transcribe, process, and respond to spoken and written queries with intelligence and adaptability. This system integrates LLMs, Web Search, RAG (PDF-based Retrieval), and Speech Recognition to create a dynamic, multi-modal assistant.The bot leverages real-time speech recognition to transcribe conversations accurately and uses context-aware retrieval to provide precise, relevant responses. By intelligently selecting between retrieved document knowledge, web search results, and LLM-generated answers, it ensures reliable and well-informed assistance. Designed for scalability, it can be expanded with additional models, APIs, and deployment options.


Features

Real-time Speech Recognition – Converts audio into text for seamless voice interaction.
RAG-Enabled Contextual Understanding – Upload PDFs to enhance AI responses with document knowledge.
Intelligent Web Search – Fetches real-time information using the Tavily API when needed.
Conversational LLM Integration – Powered by Llama 3.1 (70B) via GroqCloud for accurate and context-aware responses.
Smart Decision Making – Dynamically chooses the best response strategy (RAG, Web Search, or LLM).
Persistent Vector Storage – Uses ChromaDB to store and retrieve PDF-based embeddings efficiently.
Streamlit-Based Interactive UI – A simple yet powerful web app interface for user interaction.


Tech Stack

Programming Language: Python
Framework: Streamlit
LLM Provider: GroqCloud (Llama 3.1 - 70B)
Embeddings & Vector Store: HuggingFace Embeddings + ChromaDB
Speech Recognition: Google SpeechRecognition API
Web Search API: Tavily API
Document Processing: PyPDFLoader
