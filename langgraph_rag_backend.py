CHUNK_SIZE = 500
CHUNK_OVERLAP = 75
TOP_K = 8
from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

import groq
import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, trim_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# -------------------
# 0. Settings (evaluation ke baad yahan values badal dena)
# -------------------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 3
MAX_HISTORY_MESSAGES = 10      # LLM ko sirf last itne messages bhejte hain
MAX_CHARS_PER_CHUNK = 1200     # tool output mein har chunk ki limit
MAX_TOOL_OUTPUT_CHARS = 4000   # rag_tool ke total context ki limit

# -------------------
# 1. LLM + embeddings
# -------------------
llm = ChatGroq(model="openai/gpt-oss-120b")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -------------------
# 2. PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and str(thread_id) in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[str(thread_id)]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": TOP_K}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass


# -------------------
# 3. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage.
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return {"error": "ALPHAVANTAGE_API_KEY is not set."}

    try:
        r = requests.get(
            "https://www.alphavantage.co/query",
            params={"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": api_key},
            timeout=10,
        )
        return r.json()
    except Exception as e:
        return {"error": str(e)}


@tool
def rag_tool(query: str, config: RunnableConfig) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Pass only the search query. The thread is detected automatically.
    """
    thread_id = (config or {}).get("configurable", {}).get("thread_id")
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    results = retriever.invoke(query)

    context = []
    pages = []
    total = 0
    for doc in results:
        text = doc.page_content[:MAX_CHARS_PER_CHUNK]
        if total + len(text) > MAX_TOOL_OUTPUT_CHARS:
            break
        total += len(text)
        context.append(text)
        page = doc.metadata.get("page")
        pages.append(page + 1 if isinstance(page, int) else None)  # 1-based page number

    return {
        "query": query,
        "context": context,
        "pages": pages,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


tools = [search_tool, get_stock_price, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)


# -------------------
# 4. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 5. Nodes
# -------------------
def chat_node(state: ChatState, config: RunnableConfig = None):
    """LLM node that may answer or request a tool call."""
    thread_id = ((config or {}).get("configurable", {}) or {}).get("thread_id")
    doc_meta = _THREAD_METADATA.get(str(thread_id)) if thread_id else None

    if doc_meta:
        doc_status = (
            f"A PDF named '{doc_meta['filename']}' HAS ALREADY BEEN UPLOADED and indexed "
            "for this chat. Never ask the user to upload it again. For any question that "
            "could relate to the document, ALWAYS call `rag_tool` first with a search "
            "query, then answer from the result and mention the page numbers."
        )
    else:
        doc_status = (
            "No PDF is indexed for this chat yet. If the user asks about a document, "
            "ask them to upload a PDF."
        )

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. "
            + doc_status
            + " You can also use the web search, stock price, and calculator tools "
            "when helpful."
        )
    )

    # History trim: sirf last N messages LLM ko bhejo (token limit se bachne ke liye)
    history = trim_messages(
        state["messages"],
        max_tokens=MAX_HISTORY_MESSAGES,
        token_counter=len,
        strategy="last",
        start_on="human",
        include_system=False,
    )
    if not history:  # safety: kuch bhi na bacha to last message bhej do
        history = state["messages"][-1:]

    messages = [system_message, *history]

    try:
        response = llm_with_tools.invoke(messages, config=config)
    except groq.APIStatusError as e:
        # Logs mein asli error dikhega, user ko app crash ki jagah message milega
        print("GROQ STATUS:", e.status_code)
        print("GROQ BODY:", e.response.text)

        if e.status_code == 429:
            text = "Groq rate limit hit ho gayi. Thodi der ruko aur dobara try karo."
        elif e.status_code == 413:
            text = "Request bahut badi thi (token limit). Chhota sawal puchho ya naya chat shuru karo."
        else:
            text = f"Groq API error (status {e.status_code}). Logs check karo."
        response = AIMessage(content=text)

    return {"messages": [response]}


tool_node = ToolNode(tools)

# -------------------
# 6. Checkpointer
# -------------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# -------------------
# 7. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


# -------------------
# 8. Helpers
# -------------------
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})