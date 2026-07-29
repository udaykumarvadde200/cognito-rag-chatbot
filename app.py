"""
app.py: Streamlit UI for Cognito LangGraph RAG Chatbot
Features:
    - Chat interface
    - PDF/TXT file upload
    - Weather, Stocks, Calculator, RAG, Web Search
"""

import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
from graph.graph import app
from graph.nodes.file_ingest import file_ingest_node
from graph.state import GraphState

load_dotenv()

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Cognito — Advanced RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Cognito — Advanced RAG Chatbot")
st.caption("Powered by LangGraph + HuggingFace | RAG · Weather · Stocks · Calculator · Web Search")

# ─────────────────────────────────────────────
# SIDEBAR — File Upload
# ─────────────────────────────────────────────

with st.sidebar:
    st.header("📁 Upload Documents")
    st.caption("Upload PDF or TXT files to chat with your documents")

    uploaded_files = st.file_uploader(
        label="Choose files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("📥 Ingest Files", type="primary"):
            with st.spinner("Ingesting files into ChromaDB..."):
                # Save uploaded files to temp directory
                temp_paths = []
                for uploaded_file in uploaded_files:
                    suffix = os.path.splitext(uploaded_file.name)[-1]
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=suffix
                    ) as tmp:
                        tmp.write(uploaded_file.read())
                        temp_paths.append(tmp.name)

                # Run file ingest node
                ingest_state: GraphState = {
                    "question": "",
                    "generation": "",
                    "web_search": False,
                    "documents": [],
                    "tool_type": None,
                    "tool_output": None,
                    "uploaded_files": temp_paths,
                    "city": None,
                    "stock_symbol": None,
                    "expression": None,
                }
                file_ingest_node(ingest_state)

                st.success(f"✅ {len(uploaded_files)} file(s) ingested successfully!")
                st.info("You can now ask questions about your uploaded documents.")

    st.divider()

    st.header("💡 What can I ask?")
    st.markdown("""
    - 📄 **Documents**: *"What is agent memory?"*
    - 🌤️ **Weather**: *"Weather in Mumbai?"*
    - 📈 **Stocks**: *"Stock price of Tesla?"*
    - 🧮 **Calculator**: *"Calculate 25 * 48 + 100"*
    - 🌐 **Web Search**: *"Latest AI news 2026?"*
    - 📁 **Uploaded files**: Upload PDF → Ask questions
    """)

    st.divider()
    st.caption("Built with LangGraph + HuggingFace + Streamlit")

# ─────────────────────────────────────────────
# CHAT HISTORY
# ─────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ─────────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────────

if question := st.chat_input("Ask me anything..."):

    # Display user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Run LangGraph
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                initial_state: GraphState = {
                    "question": question,
                    "generation": "",
                    "web_search": False,
                    "documents": [],
                    "tool_type": None,
                    "tool_output": None,
                    "uploaded_files": [],
                    "city": None,
                    "stock_symbol": None,
                    "expression": None,
                }

                result = app.invoke(input=initial_state)
                answer = result.get("generation", "Sorry, I could not generate an answer.")
                tool_type = result.get("tool_type", None)

                # Show tool badge
                tool_badges = {
                    "weather":    "🌤️ Weather",
                    "stocks":     "📈 Stocks",
                    "calculator": "🧮 Calculator",
                }
                if tool_type in tool_badges:
                    st.caption(f"Source: {tool_badges[tool_type]}")
                elif result.get("web_search"):
                    st.caption("Source: 🌐 Web Search")
                else:
                    st.caption("Source: 📄 Documents")

                st.markdown(answer)

            except Exception as e:
                answer = f"❌ Error: {str(e)}"
                st.error(answer)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})