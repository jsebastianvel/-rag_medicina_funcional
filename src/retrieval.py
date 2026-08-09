"""Retrieval hibrido: combina busqueda densa (embeddings) con BM25 (sparse)
mediante Reciprocal Rank Fusion (RRF), y luego reordena los candidatos
fusionados con un cross-encoder (reranking).

Por que hibrido: la busqueda densa entiende significado (sinonimos,
parafrasis) pero puede fallar con terminos exactos poco frecuentes; BM25 es
al reves, fuerte en coincidencias literales de palabras pero ciego al
significado. Combinar ambas cubre los puntos ciegos de cada una. El
reranking final con un cross-encoder es mas lento pero mas preciso que los
embeddings porque compara la query y el chunk juntos en vez de por separado,
asi que solo se aplica sobre los pocos candidatos ya filtrados.
"""

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.config import TOP_K_DENSE, TOP_K_SPARSE, TOP_K_FINAL, RERANKER_MODEL_NAME
from src.vector_store import load_index, dense_search, load_chunks

RRF_K = 60

_bm25 = None
_bm25_chunks = None
_reranker = None


def _tokenize(text):
    return text.lower().split()


def get_bm25():
    global _bm25, _bm25_chunks
    if _bm25 is None:
        chunks = load_chunks()
        tokenized = [_tokenize(c["text"]) for c in chunks]
        _bm25 = BM25Okapi(tokenized)
        _bm25_chunks = chunks
    return _bm25, _bm25_chunks


def sparse_search(query, k):
    bm25, chunks = get_bm25()
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    results = []
    for idx in ranked[:k]:
        results.append({
            "score": float(scores[idx]),
            "chunk": chunks[idx],
        })
    return results


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker


def reciprocal_rank_fusion(dense_results, sparse_results, rrf_k=RRF_K):
    """Combina dos rankings sin necesitar normalizar sus escalas de score:
    cada chunk suma 1 / (rrf_k + posicion) por cada lista en la que aparece.
    Un chunk que sale bien ubicado en ambas busquedas termina arriba."""
    fused_scores = {}
    chunk_by_id = {}

    for rank, item in enumerate(dense_results):
        chunk_id = item["chunk"]["id"]
        chunk_by_id[chunk_id] = item["chunk"]
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank + 1)

    for rank, item in enumerate(sparse_results):
        chunk_id = item["chunk"]["id"]
        chunk_by_id[chunk_id] = item["chunk"]
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank + 1)

    ranked_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)
    return [{"score": fused_scores[cid], "chunk": chunk_by_id[cid]} for cid in ranked_ids]


def rerank(query, candidates, k):
    reranker = get_reranker()
    pairs = [[query, candidate["chunk"]["text"]] for candidate in candidates]
    scores = reranker.predict(pairs)

    scored_pairs = list(zip(scores, candidates))
    scored_pairs.sort(key=lambda pair: pair[0], reverse=True)

    results = []
    for score, item in scored_pairs[:k]:
        results.append({
            "score": float(score),
            "chunk": item["chunk"],
        })
    return results


def hybrid_search(query, k=TOP_K_FINAL):
    index, chunks = load_index()
    dense_results = dense_search(query, TOP_K_DENSE, index, chunks)
    sparse_results = sparse_search(query, TOP_K_SPARSE)

    fused = reciprocal_rank_fusion(dense_results, sparse_results)
    return rerank(query, fused, k)


if __name__ == "__main__":
    query = "que es la inflamacion cronica de bajo grado"

    print("Query: " + query)

    print("")
    print("=== Solo dense ===")
    index, chunks = load_index()
    for item in dense_search(query, 3, index, chunks):
        print(str(round(item["score"], 3)) + " -- " + item["chunk"]["title"])

    print("")
    print("=== Solo BM25 (sparse) ===")
    for item in sparse_search(query, 3):
        print(str(round(item["score"], 3)) + " -- " + item["chunk"]["title"])

    print("")
    print("=== Hibrido + reranking (resultado final) ===")
    for item in hybrid_search(query, TOP_K_FINAL):
        print(str(round(item["score"], 3)) + " -- " + item["chunk"]["title"])
        print(item["chunk"]["text"][:150] + "...")
        print("")