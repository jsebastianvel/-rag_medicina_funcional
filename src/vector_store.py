"""Indice vectorial local basado en FAISS.

FAISS solo guarda vectores; el texto y los metadatos de cada chunk viven en
data/index/chunks.json (ver ingest.py). El indice y el archivo de chunks se
relacionan por posicion: el vector en la fila i del indice corresponde al
chunk con indice i en la lista cargada desde chunks.json.

Nota: se serializa el indice a bytes y se escribe con open() de Python en vez
de con faiss.write_index/read_index. En este entorno, cargar PyTorch antes
altera el locale del proceso y rompe el fopen nativo de FAISS para rutas con
espacios; escribir los bytes nosotros mismos evita ese problema.
"""

import json

import faiss
import numpy as np

from src.config import CHUNKS_PATH, FAISS_INDEX_PATH
from src.embeddings import embed_texts


def load_chunks():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_index(index, path):
    data = faiss.serialize_index(index)
    with open(path, "wb") as fh:
        fh.write(data.tobytes())


def _load_index_from_disk(path):
    with open(path, "rb") as fh:
        data = np.frombuffer(fh.read(), dtype="uint8")
    return faiss.deserialize_index(data)


def build_index():
    chunks = load_chunks()
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)

    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(np.array(vectors, dtype="float32"))

    _save_index(index, FAISS_INDEX_PATH)
    return index, chunks


def load_index():
    index = _load_index_from_disk(FAISS_INDEX_PATH)
    chunks = load_chunks()
    return index, chunks


def dense_search(query, k, index=None, chunks=None):
    if index is None or chunks is None:
        index, chunks = load_index()

    query_vector = embed_texts([query])
    query_vector = np.array(query_vector, dtype="float32")

    scores, indices = index.search(query_vector, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx]
        results.append({
            "score": float(score),
            "chunk": chunk,
        })
    return results


if __name__ == "__main__":
    print("Construyendo indice FAISS a partir de los chunks...")
    index, chunks = build_index()
    print(str(index.ntotal) + " vectores indexados (dimension " + str(index.d) + ")")

    query = "que es la inflamacion cronica de bajo grado"
    print("")
    print("Query de prueba: " + query)
    results = dense_search(query, 3, index, chunks)
    for r in results:
        print("")
        print("score=" + str(round(r["score"], 3)) + " -- " + r["chunk"]["title"])
        print(r["chunk"]["text"][:150] + "...")