# LangGraph Multi Utility Chatbot

A multi-utility chatbot built with LangGraph, Groq and Streamlit.

## Features
- PDF RAG: upload a PDF and ask questions about it
- Tool calling
- Persistent chat threads (SQLite)
- Multiple chat sessions

## Setup

1. Clone the repo
```bash
   git clone https://github.com/manish1934/langgraph-multi-utility-chatbot.git
   cd langgraph-multi-utility-chatbot
```

2. Install dependencies
```bash
   pip install -r requirements.txt
```

3. Create a `.env` file (copy `.env.example`) and add your API keys

4. Run the app
```bash
   streamlit run streamlit_rag_frontend.py
