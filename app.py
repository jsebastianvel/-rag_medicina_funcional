"""Interfaz de chat para el sistema RAG, usando Gradio.

Cada pregunta pasa por el pipeline completo: retrieval hibrido con
reranking (src/retrieval.py) y generacion con citas (src/generate.py).
answer_question adapta ese resultado al formato que espera
gr.ChatInterface (recibe el mensaje y el historial, devuelve texto)."""

import gradio as gr

from src.generate import generate_answer

NEWLINE = chr(10)

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
    demo.launch()