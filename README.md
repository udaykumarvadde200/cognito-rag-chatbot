
# 🤖 Cognito — Advanced RAG Chatbot

An **Agentic CRAG (Corrective RAG) system** built with **LangGraph + HuggingFace Mistral-7B** that answers questions from multiple sources — documents, real-time weather, live stock prices, calculator, and web search.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📄 **Document Q&A** | Ask questions about research articles using RAG pipeline |
| 🌤️ **Real-time Weather** | Get current weather for any city via OpenWeatherMap |
| 📈 **Live Stock Prices** | Fetch live stock data for any company via yFinance |
| 🧮 **Calculator** | Evaluate math expressions instantly |
| 🌐 **Web Search Fallback** | Falls back to Tavily web search when documents don't have the answer |
| 📁 **File Upload** | Upload your own PDF/TXT and chat with it |
| 🔄 **Self-Reflection (CRAG)** | Grades retrieved documents and corrects bad retrieval automatically |

---

## 🏗️ Architecture

```
User Question
      ↓
Smart Router (keyword + LLM based)
      ↓
┌──────────────────────────────────────────────┐
│  Weather  Stocks  Calculator  RAG  WebSearch │
└──────────────────────────────────────────────┘
      ↓
Generate Answer (Mistral-7B via HuggingFace)
      ↓
Self-Reflection Check
(Is answer grounded? Does it address the question?)
      ↓
Final Answer ✅
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Agent Framework** | LangGraph |
| **LLM (Generation + Grading + Routing)** | HuggingFace Mistral-7B-Instruct-v0.3 (Free) |
| **Embeddings** | HuggingFace BAAI/bge-small-en-v1.5 (Free, CPU) |
| **Vector Store** | ChromaDB (local, persisted at `./.chroma`) |
| **Weather API** | OpenWeatherMap (Free) |
| **Stock Data** | yFinance (Free, no API key) |
| **Web Search** | Tavily (Free tier) |
| **UI** | Streamlit |
| **Document Loaders** | LangChain Community (WebBaseLoader, PyPDFLoader, TextLoader) |
| **Text Splitting** | RecursiveCharacterTextSplitter (chunk_size=500, overlap=50) |

---

## 🤗 HuggingFace Models Used

| Purpose | Model |
|---------|-------|
| **LLM — Generation** | `mistralai/Mistral-7B-Instruct-v0.3` |
| **LLM — Retrieval Grader** | `mistralai/Mistral-7B-Instruct-v0.3` |
| **LLM — Hallucination Grader** | `mistralai/Mistral-7B-Instruct-v0.3` |
| **LLM — Answer Grader** | `mistralai/Mistral-7B-Instruct-v0.3` |
| **LLM — Router** | `mistralai/Mistral-7B-Instruct-v0.3` |
| **Embeddings** | `BAAI/bge-small-en-v1.5` |

> ✅ No OpenAI API key needed. Everything runs on free HuggingFace APIs.

---

## 📁 Project Structure

```
cognito/
├── graph/
│   ├── chains/
│   │   ├── __init__.py
│   │   ├── answer_grader.py        # Grades if answer addresses question
│   │   ├── generation.py           # LLM generation chain (Mistral-7B)
│   │   ├── hallucination_grader.py # Checks if answer is grounded in facts
│   │   ├── retrieval_grader.py     # Grades document relevance
│   │   └── router.py               # Routes to vectorstore or websearch
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── generate.py             # Generate node (handles tool + RAG output)
│   │   ├── grade_documents.py      # Grade documents node
│   │   ├── retrieve.py             # ChromaDB retrieval node
│   │   ├── web_search.py           # Tavily web search node
│   │   ├── weather.py              # Weather tool node (OpenWeatherMap)
│   │   ├── stocks.py               # Stock price tool node (yFinance)
│   │   ├── calculator.py           # Calculator tool node (safe eval)
│   │   └── file_ingest.py          # PDF/TXT upload + ChromaDB ingestion
│   ├── __init__.py
│   ├── consts.py                   # Node name constants
│   ├── graph.py                    # LangGraph workflow (nodes + edges + routing)
│   └── state.py                    # GraphState TypedDict definition
├── ingestion.py                    # URL ingestion + ChromaDB setup + get_retriever()
├── app.py                          # Streamlit UI (chat + file upload sidebar)
├── main.py                         # Terminal entry point (5 test cases)
├── requirements.txt                # All dependencies
├── .env.example                    # API key template
├── .gitignore
└── README.md
```

---

## ⚙️ How Ingestion Works

### URL Ingestion (default — runs once on first launch)
```python
# ingestion.py
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",   # free HuggingFace embedding model
    model_kwargs={"device": "cpu"},          # runs on CPU — no GPU needed
    encode_kwargs={"normalize_embeddings": True},
)

# Auto-ingests 5 research article URLs into ChromaDB on first run
# Skips ingestion if ChromaDB already exists (no re-indexing every run)
```

### File Upload Ingestion (user-triggered via sidebar)
```python
# graph/nodes/file_ingest.py
# Supports: .pdf (PyPDFLoader) and .txt (TextLoader)
# Chunks: 500 chars, 50 overlap
# Embeds with same BAAI/bge-small-en-v1.5 model
# Adds to existing ChromaDB collection — same index as URL docs
```

### Retriever (always fresh)
```python
# ingestion.py
def get_retriever():
    # Called fresh every time retrieve node runs
    # Automatically picks up newly uploaded file chunks
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},   # returns top 4 relevant chunks
    )
```

---

## 🔄 CRAG Flow (Corrective RAG)

```
1.  User asks question
2.  Router detects intent:
      weather/stocks/calculator keywords → tool node directly
      everything else → LLM router (vectorstore vs websearch)
3.  If RAG: retrieve top-4 chunks from ChromaDB
4.  Grade each chunk — relevant or not?
5.  If any chunk not relevant → trigger web search fallback
6.  Generate answer from context using Mistral-7B
7.  Hallucination check — is answer grounded in the context?
8.  If not grounded → retry generation (loop back)
9.  Answer check — does it actually address the question?
10. If not useful → trigger web search again (loop back)
11. Return final answer ✅
```

---

## 🚀 Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/cognito-rag-chatbot.git
cd cognito-rag-chatbot
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up API Keys
```bash
cp .env.example .env
```

Fill in your `.env` file:
```env
HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
OPENWEATHER_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
```

### 5. Get Free API Keys

| API | Link | Cost |
|-----|------|------|
| **HuggingFace** | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | ✅ Free (Read token) |
| **OpenWeatherMap** | [openweathermap.org/api](https://openweathermap.org/api) | ✅ Free (1000 calls/day) |
| **Tavily** | [tavily.com](https://tavily.com) | ✅ Free (1000 searches/month) |

### 6. Run the App
```bash
# Streamlit UI (recommended)
python -m streamlit run app.py

# OR terminal version
python main.py
```

---

## 💬 Example Questions

| Type | Example |
|------|---------|
| 📄 Documents | *"What is agent memory?"* |
| 🌤️ Weather | *"What is the weather in Mumbai?"* |
| 📈 Stocks | *"What is the stock price of Tesla?"* |
| 🧮 Calculator | *"Calculate 25 * 48 + 100"* |
| 🌐 Web Search | *"Latest AI news in 2026?"* |
| 📁 Your files | Upload PDF → *"Summarize this document"* |

---

## 📦 Key Dependencies

```txt
langchain
langchain-core
langchain-community
langchain-chroma
langchain-huggingface
langchain-text-splitters
langgraph
huggingface-hub
sentence-transformers
transformers
chromadb
pypdf
yfinance
tavily-python
streamlit
python-dotenv
```

---

## 👨‍💻 Built By

**Vadde Uday Kumar**
Final Year ECE Student — RGUKT RK Valley
Summer Internship Project 2026

---

## 📄 License

MIT License — free to use and modify.
