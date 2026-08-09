"""Carga los documentos de data/raw/, los divide en chunks con solapamiento
y guarda el resultado en data/index/chunks.json."""

import json
import os

from src.config import DATA_RAW_DIR, CHUNKS_PATH, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS

SENTENCE_ENDERS = (".", "!", "?")


def _parse_document(path):
    with open(path, "r", encoding="utf-8-sig") as fh:
        raw = fh.read()

    header, _, body = raw.partition("---")
    title = ""
    source = ""
    for line in header.splitlines():
        if line.startswith("TITULO:"):
            title = line.split(":", 1)[1].strip()
        elif line.startswith("FUENTE:"):
            source = line.split(":", 1)[1].strip()

    return title, source, body.strip()


def _split_paragraphs(text):
    paragraphs = []
    current_lines = []
    for line in text.splitlines():
        if line.strip() == "":
            if current_lines:
                paragraphs.append(" ".join(current_lines).strip())
                current_lines = []
        else:
            current_lines.append(line.strip())
    if current_lines:
        paragraphs.append(" ".join(current_lines).strip())
    return paragraphs


def _split_sentences(paragraph):
    sentences = []
    current = ""
    for ch in paragraph:
        current += ch
        if ch in SENTENCE_ENDERS:
            sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())
    return sentences


def _split_long_paragraph(paragraph, max_words):
    """Si un parrafo por si solo excede el tamano maximo, lo parte por oraciones."""
    sentences = _split_sentences(paragraph)
    chunks = []
    current = []
    current_len = 0
    for sentence in sentences:
        n = len(sentence.split())
        if current and current_len + n > max_words:
            chunks.append(" ".join(current))
            current = []
            current_len = 0
        current.append(sentence)
        current_len += n
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_text(text, chunk_size=CHUNK_SIZE_WORDS, overlap=CHUNK_OVERLAP_WORDS):
    """Divide el texto en chunks respetando parrafos cuando es posible.

    Estrategia: acumula parrafos completos hasta acercarse al limite de
    palabras. Un parrafo nunca se corta a la mitad, salvo que el parrafo por
    si solo ya exceda el limite (entonces se corta por oraciones). Cada chunk
    nuevo arranca repitiendo las ultimas overlap palabras del chunk
    anterior, para no perder contexto en el borde de un corte.
    """
    paragraphs = []
    for p in _split_paragraphs(text):
        if len(p.split()) > chunk_size:
            paragraphs.extend(_split_long_paragraph(p, chunk_size))
        else:
            paragraphs.append(p)

    chunks = []
    current_words = []

    for paragraph in paragraphs:
        paragraph_words = paragraph.split()
        if current_words and len(current_words) + len(paragraph_words) > chunk_size:
            chunks.append(" ".join(current_words))
            overlap_words = current_words[-overlap:] if overlap else []
            current_words = overlap_words.copy()
        current_words.extend(paragraph_words)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def build_chunks():
    """Recorre data/raw/, chunkea cada documento y guarda todo en chunks.json."""
    all_chunks = []
    chunk_id = 0

    filenames = sorted(name for name in os.listdir(DATA_RAW_DIR) if name.endswith(".txt"))
    for filename in filenames:
        path = os.path.join(DATA_RAW_DIR, filename)
        title, source, body = _parse_document(path)
        text_chunks = chunk_text(body)

        for text in text_chunks:
            all_chunks.append({
                "id": chunk_id,
                "text": text,
                "title": title,
                "source": source,
                "filename": filename,
            })
            chunk_id += 1

    os.makedirs(os.path.dirname(CHUNKS_PATH), exist_ok=True)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as fh:
        json.dump(all_chunks, fh, ensure_ascii=False, indent=2)

    return all_chunks


if __name__ == "__main__":
    chunks = build_chunks()
    n_docs = len(set(c["filename"] for c in chunks))
    print(str(len(chunks)) + " chunks generados a partir de " + str(n_docs) + " documentos")
    for c in chunks[:3]:
        n_words = len(c["text"].split())
        print("")
        print("--- chunk " + str(c["id"]) + " (" + c["filename"] + ", " + str(n_words) + " palabras) ---")
        print(c["text"][:200] + "...")