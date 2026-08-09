"""Genera embeddings locales de texto usando un modelo sentence-transformers.

No se llama a ninguna API externa: el modelo se descarga una sola vez la
primera vez que se usa y despues corre enteramente en tu maquina (CPU).
"""

from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL_NAME

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts):
    """Convierte una lista de strings en una matriz numpy de vectores unitarios
    (normalizados), lista para comparar por producto punto = similitud coseno."""
    model = get_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)