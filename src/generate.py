"""Genera la respuesta final combinando los chunks recuperados con un LLM
(Gemini), pidiendole que cite de que documento sale cada afirmacion y que
no invente informacion fuera del contexto entregado."""

import os

from dotenv import load_dotenv
from google import genai

from src.config import GENERATION_MODEL_NAME
from src.retrieval import hybrid_search

load_dotenv()

NEWLINE = chr(10)

PROMPT_TEMPLATE = """Eres un asistente que responde preguntas sobre medicina funcional usando UNICAMENTE la informacion del contexto de abajo. No inventes datos que no esten en el contexto. Si el contexto no alcanza para responder, dilo explicitamente en vez de adivinar.

Al final de cada afirmacion, cita entre corchetes el numero de la fuente que la respalda, por ejemplo [Fuente 1].

CONTEXTO:
{context}

PREGUNTA: {query}

RESPUESTA (en espanol, citando fuentes):"""


def _get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY no configurada en el entorno (.env)")
    return genai.Client(api_key=api_key)


def _build_context(results):
    blocks = []
    for position, item in enumerate(results):
        chunk = item["chunk"]
        label = "Fuente " + str(position + 1) + " (" + chunk["filename"] + ")"
        blocks.append(label + ":" + NEWLINE + chunk["text"])
    return (NEWLINE + NEWLINE).join(blocks)


def generate_answer(query, k=None):
    results = hybrid_search(query, k) if k else hybrid_search(query)
    context_text = _build_context(results)
    prompt = PROMPT_TEMPLATE.format(context=context_text, query=query)

    client = _get_client()
    response = client.models.generate_content(model=GENERATION_MODEL_NAME, contents=prompt)

    sources = []
    for position, item in enumerate(results):
        chunk = item["chunk"]
        sources.append({
            "label": "Fuente " + str(position + 1),
            "title": chunk["title"],
            "source": chunk["source"],
            "filename": chunk["filename"],
        })

    return {
        "query": query,
        "answer": response.text,
        "sources": sources,
    }


if __name__ == "__main__":
    query = "que es la inflamacion cronica de bajo grado y como se relaciona con la dieta"

    result = generate_answer(query)

    print("Pregunta: " + result["query"])
    print("")
    print("Respuesta:")
    print(result["answer"])
    print("")
    print("Fuentes citadas:")
    for source in result["sources"]:
        print(source["label"] + ": " + source["title"] + " (" + source["source"] + ")")