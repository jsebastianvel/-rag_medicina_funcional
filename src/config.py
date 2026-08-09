"""Configuracion central del proyecto: rutas y parametros del pipeline."""

import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
DATA_INDEX_DIR = os.path.join(ROOT_DIR, "data", "index")

CHUNKS_PATH = os.path.join(DATA_INDEX_DIR, "chunks.json")

# RAG_FAISS_INDEX_PATH permite redirigir el archivo del indice a otra carpeta
# (por ejemplo un directorio temporal) si el destino por defecto no es
# escribible en el entorno actual. Por defecto vive dentro del proyecto.
FAISS_INDEX_PATH = os.environ.get(
    "RAG_FAISS_INDEX_PATH",
    os.path.join(DATA_INDEX_DIR, "faiss.index"),
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
GENERATION_MODEL_NAME = "gemini-flash-latest"

TOP_K_DENSE = 8
TOP_K_SPARSE = 8
TOP_K_FINAL = 4