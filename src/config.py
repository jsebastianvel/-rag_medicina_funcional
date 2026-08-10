"""Configuracion central del proyecto: rutas y parametros del pipeline."""

import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
DATA_INDEX_DIR = os.path.join(ROOT_DIR, "data", "index")

# RAG_INDEX_DIR permite redirigir chunks.json y faiss.index a otra carpeta
# (por ejemplo fuera de una carpeta sincronizada con OneDrive, donde escribir
# archivos nuevos puede fallar) si el destino por defecto no es escribible en
# el entorno actual. Por defecto ambos viven dentro del proyecto.
_INDEX_DIR = os.environ.get("RAG_INDEX_DIR", DATA_INDEX_DIR)
CHUNKS_PATH = os.path.join(_INDEX_DIR, "chunks.json")

# RAG_FAISS_INDEX_PATH permite redirigir solo el indice (compatibilidad con
# el workaround anterior); RAG_INDEX_DIR cubre ambos archivos si se prefiere
# fijar un solo directorio externo.
FAISS_INDEX_PATH = os.environ.get(
    "RAG_FAISS_INDEX_PATH",
    os.path.join(_INDEX_DIR, "faiss.index"),
)

# Chunking: tamanos en palabras (no caracteres) para que el limite sea
# independiente de la longitud de las palabras del idioma.
CHUNK_SIZE_WORDS = 150
CHUNK_OVERLAP_WORDS = 30

# Modelo de embeddings local (multilingue, soporta espanol). Se descarga una
# sola vez la primera vez que se usa.
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Cross-encoder multilingue para reranking (tambien local).
RERANKER_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# LLM para la generacion final (via API, no local).
# "gemini-flash-latest" tiene solo 20 solicitudes/dia gratis en esta cuenta;
# "gemini-flash-lite-latest" tiene un limite mas usable (15/minuto, 500/dia).
GENERATION_MODEL_NAME = "gemini-flash-lite-latest"

TOP_K_DENSE = 8
TOP_K_SPARSE = 8
TOP_K_FINAL = 4