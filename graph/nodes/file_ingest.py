from typing import Any, Dict, List
import os
import tempfile
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from graph.state import GraphState

# Same embedding model as ingestion.py
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
CHROMA_DIR = "chroma_db"


def file_ingest_node(state: GraphState) -> Dict[str, Any]:
    """
    Load uploaded PDF/TXT files, chunk them,
    embed and store in ChromaDB, then pass to RETRIEVE.
    """
    print("---file ingest---")
    uploaded_files = state.get("uploaded_files", [])
    question = state["question"]

    if not uploaded_files:
        print("---no files to ingest---")
        return {"question": question, "documents": []}

    all_docs: List[Document] = []

    for file_path in uploaded_files:
        print(f"---ingesting file: {file_path}---")
        try:
            ext = os.path.splitext(file_path)[-1].lower()
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
            elif ext == ".txt":
                loader = TextLoader(file_path)
            else:
                print(f"---unsupported file type: {ext}, skipping---")
                continue

            docs = loader.load()
            all_docs.extend(docs)

        except Exception as e:
            print(f"---failed to load {file_path}: {str(e)}---")
            continue

    if not all_docs:
        return {"question": question, "documents": []}

    # Chunk documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(all_docs)
    print(f"---{len(chunks)} chunks created from uploaded files---")

    # Add to existing ChromaDB
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    vectorstore.add_documents(chunks)
    print("---chunks added to ChromaDB---")

    return {
        "question":       question,
        "documents":      [],
        "uploaded_files": uploaded_files,
    }