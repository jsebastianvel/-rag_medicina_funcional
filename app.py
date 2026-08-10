"""Interfaz de chat para el sistema RAG, usando Gradio.

Cada pregunta pasa por el pipeline completo: retrieval hibrido con
reranking (src/retrieval.py) y generacion con citas (src/generate.py).
answer_question adapta ese resultado al formato que espera
gr.ChatInterface (recibe el mensaje y el historial, devuelve texto)."""

import os

# sentence-transformers (torch) must be imported before gradio - importing
# gradio first causes torch's later import to hang indefinitely on this
# machine (reproduced reliably; order below is the fix, not a style choice).
from src.config import CHUNKS_PATH, FAISS_INDEX_PATH
from src.generate import generate_answer
from src.ingest import build_chunks
from src.vector_store import build_index

import gradio as gr

NEWLINE = chr(10)


def ensure_index_built():
    """Hosting platforms like Hugging Face Spaces rebuild the container (and
    its filesystem) on every deploy, so the FAISS index is missing on first
    boot - chunks.json and faiss.index are gitignored, not committed. Building
    them lazily here makes deploys self-healing without a separate "publish
    index" step."""
    if os.path.exists(CHUNKS_PATH) and os.path.exists(FAISS_INDEX_PATH):
        return
    print("Indice no encontrado, construyendolo (puede tardar un momento)...")
    build_chunks()
    build_index()
    print("Indice construido.")


EXAMPLES = [
    "Que es la medicina funcional?",
    "Que es la inflamacion cronica de bajo grado?",
    "Cuales son las fases de la detoxificacion hepatica?",
    "Que relacion hay entre la microbiota y la obesidad?",
]


def _format_sources(sources):
    lines = []
    for source in sources:
        lines.append(source["label"] + ": " + source["title"] + " (" + source["source"] + ")")
    return NEWLINE.join(lines)


def answer_question(message, history):
    result = generate_answer(message)
    sources_text = _format_sources(result["sources"])
    return result["answer"] + NEWLINE + NEWLINE + "---" + NEWLINE + "**Fuentes:**" + NEWLINE + sources_text


demo = gr.ChatInterface(
    fn=answer_question,
    title="RAG Medicina Funcional",
    description=(
        "Preguntas respondidas con un pipeline de RAG puro: retrieval hibrido "
        "(embeddings + BM25 + reranking) sobre un corpus de medicina funcional, "
        "y generacion con citas via Gemini."
    ),
    examples=EXAMPLES,
)


if __name__ == "__main__":
    ensure_index_built()
    # SPACE_ID is set by Hugging Face Spaces; don't try to open a local
    # browser inside that headless container.
    demo.launch(inbrowser=os.environ.get("SPACE_ID") is None)