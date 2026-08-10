"""Interfaz de chat para el sistema RAG, usando Streamlit (deploy target:
Streamlit Community Cloud - Hugging Face Spaces dropped free compute for
Gradio/Docker Spaces, only Static remains free there, which can't run this
app's Python backend).

Mismo pipeline que app.py (Gradio, uso local): retrieval hibrido con
reranking (src/retrieval.py) y generacion con citas (src/generate.py)."""

import os

# Import order precaution: a real bug was found where importing a UI
# framework (gradio) before sentence-transformers/torch causes torch's
# import to hang indefinitely on this machine. Untested with streamlit, but
# importing src.* first here too as a precaution.
from src.config import CHUNKS_PATH, FAISS_INDEX_PATH
from src.generate import generate_answer
from src.ingest import build_chunks
from src.vector_store import build_index

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="RAG Medicina Funcional", page_icon="🩺")
st.title("RAG Medicina Funcional")
st.caption(
    "Pipeline de RAG puro (sin frameworks de agentes): retrieval hibrido "
    "(embeddings locales + BM25 + reranking) y generacion con citas via Gemini."
)


@st.cache_resource
def ensure_index_built() -> bool:
    """Streamlit Community Cloud rebuilds the container (and its filesystem)
    on every deploy, so the FAISS index is missing on first boot - chunks.json
    and faiss.index are gitignored, not committed. Building them lazily here
    makes deploys self-healing. @st.cache_resource runs this once per server
    process, not on every rerun."""
    if os.path.exists(CHUNKS_PATH) and os.path.exists(FAISS_INDEX_PATH):
        return True
    build_chunks()
    build_index()
    return True


with st.spinner("Preparando el indice..."):
    ensure_index_built()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Preguntame sobre medicina funcional...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            result = generate_answer(query)
        answer = result["answer"]
        st.markdown(answer)

        sources = result.get("sources", [])
        if sources:
            with st.expander("Fuentes"):
                for source in sources:
                    st.markdown(f"- {source['label']}: {source['title']} ({source['source']})")

    st.session_state.messages.append({"role": "assistant", "content": answer})
