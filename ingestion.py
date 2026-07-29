"""
ingestion.py: Load articles/PDFs/TXTs to documents, chunk, embed, store in ChromaDB vectorstore.
Supports:
    - URL-based ingestion (default)
    - PDF/TXT file upload ingestion (via file_ingest node)
    - get_retriever() — always returns latest retriever including uploaded files
"""

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

CHROMA_DIR        = "./.chroma"
COLLECTION_NAME   = "rag-chroma"
EMBEDDING_MODEL   = "BAAI/bge-small-en-v1.5"

# Default URLs to load at startup
urls = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    "https://lilianweng.github.io/posts/2024-02-05-human-data-quality/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
]

# ─────────────────────────────────────────────
# EMBEDDINGS (HuggingFace — replaces OpenAI)
# ─────────────────────────────────────────────

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},   # change to "cuda" if you want GPU
    encode_kwargs={"normalize_embeddings": True},
)

# ─────────────────────────────────────────────
# INITIAL INGESTION (URLs → ChromaDB)
# Only runs when ChromaDB doesn't exist yet
# ─────────────────────────────────────────────

def ingest_urls():
    """Load URLs, chunk, embed and store in ChromaDB."""
    print("---ingesting URLs into ChromaDB---")
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250,
        chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    Chroma.from_documents(
        documents=doc_splits,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"---{len(doc_splits)} chunks stored in ChromaDB---")


# ─────────────────────────────────────────────
# AUTO-INGEST ON FIRST RUN
# ─────────────────────────────────────────────

if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
    print("---ChromaDB not found, running initial ingestion---")
    ingest_urls()
else:
    print("---ChromaDB found, skipping URL ingestion---")


# ─────────────────────────────────────────────
# get_retriever() — always returns latest
# ─────────────────────────────────────────────

def get_retriever():
    """
    Returns the latest ChromaDB retriever.
    Called every time retrieve node runs —
    ensures uploaded files are always included.
    """
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},  # return top 4 relevant chunks
    )