---
title: RAG Medicina Funcional
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.22.0
app_file: app.py
pinned: false
license: mit
---

# RAG Medicina Funcional

Proyecto educativo de Retrieval-Augmented Generation (RAG) "puro": cada etapa del
pipeline esta implementada de forma explicita (sin frameworks tipo LangChain) para
entender el mecanismo completo, no solo llamar una API.

Dominio: medicina funcional (corpus propio en `data/raw/`, 8 documentos reunidos de
fuentes publicas en espanol).

**Demo en vivo:** _pendiente de desplegar en Hugging Face Spaces._

## Pipeline

1. **Ingesta y chunking** (`src/ingest.py`) - carga los .txt de `data/raw/` y los
   divide en fragmentos con solapamiento.
2. **Embeddings locales** (`src/embeddings.py`) - modelo `sentence-transformers`
   multilingue, corre en tu maquina, sin llamadas a APIs externas.
3. **Vector store** (`src/vector_store.py`) - indice FAISS local + metadatos en JSON.
4. **Retrieval hibrido** (`src/retrieval.py`) - combina busqueda densa (embeddings)
   con BM25 (sparse), y aplica un cross-encoder para reranking.
5. **Generacion con citas** (`src/generate.py`) - LLM (Gemini) redacta la respuesta
   citando de que documento sale cada afirmacion.
6. **Evaluacion** (`eval/`) - set de preguntas de prueba + metricas de retrieval
   (precision y recall aproximado) y de fidelidad de la respuesta generada.

## Instalacion

### Requisitos

- Python 3.10 o superior
- pip
- Una API key gratuita de Gemini: se obtiene en Google AI Studio (aistudio.google.com,
  seccion API keys)

### 1. Entorno virtual (opcional pero recomendado)

Windows, PowerShell:

    python -m venv venv
    venv\Scripts\Activate.ps1

Windows, cmd.exe:

    python -m venv venv
    venv\Scripts\activate.bat

Si preferis no usar entorno virtual, los comandos siguientes tambien funcionan
instalando en el Python del sistema (pip usara instalacion por usuario si el
directorio global no es escribible).

### 2. Instalar dependencias

    pip install -r requirements.txt

Esto instala `sentence-transformers`, que a su vez trae PyTorch como dependencia.
La primera instalacion puede tardar varios minutos y descargar varios cientos de
MB; es normal.

### 3. Configurar la API key

Copiar la plantilla de entorno y completar la key real:

    copy .env.example .env

Abrir `.env` y reemplazar `tu_api_key_aqui` por tu API key de Gemini.

### 4. Construir el pipeline (primera vez)

Ejecutar en este orden. Cada paso depende de archivos generados por el anterior:

    python -m src.ingest
    python -m src.vector_store
    python -m src.retrieval
    python -m src.generate
    python -m eval.evaluate

Que hace cada uno:

- `src.ingest`: chunkea los documentos de `data/raw/` y escribe
  `data/index/chunks.json`.
- `src.vector_store`: genera los embeddings de cada chunk y construye el indice
  FAISS en `data/index/faiss.index`. Descarga el modelo de embeddings la primera
  vez que corre (unos cientos de MB).
- `src.retrieval`: corre una query de ejemplo por busqueda densa, por BM25, y por
  el pipeline hibrido con reranking, para comparar los tres resultados. Descarga
  el modelo de reranking (cross-encoder) la primera vez.
- `src.generate`: corre una query de ejemplo de punta a punta, incluyendo la
  llamada a la API de Gemini, y muestra la respuesta con sus fuentes citadas.
- `eval.evaluate`: corre las 10 preguntas de `eval/questions.json` contra el
  retrieval y reporta hit rate y precision; ademas prueba la generacion completa
  sobre una muestra de 3 preguntas.

Las queries de `src.retrieval` y `src.generate` estan fijas en el codigo (variable
`query` dentro de cada bloque `if __name__ == "__main__"`). Para probar tus propias
preguntas, edita esa linea o llama a las funciones `hybrid_search()` o
`generate_answer()` desde tu propio script.

## Limites a tener en cuenta

- Los modelos de embeddings y de reranking son locales: se descargan una sola vez
  y despues no necesitan conexion a internet.
- `src.generate` si llama a la API de Gemini en cada ejecucion. El modelo usado
  (`gemini-flash-lite-latest`) tiene un limite gratuito de 15 solicitudes/minuto
  y 500/dia. Si aparece un error 429 RESOURCE_EXHAUSTED, esperar unos minutos o
  al reinicio diario de la cuota.

## Problemas conocidos en Windows

- **Escritura de archivos nuevos dentro del proyecto**: si este repo vive en una
  carpeta sincronizada con OneDrive (por ejemplo bajo `Documents`), escribir
  `data/index/chunks.json` o `data/index/faiss.index` puede fallar con
  `FileNotFoundError` aunque la carpeta exista - OneDrive intercepta la escritura
  de archivos nuevos. Workaround: fijar la variable de entorno `RAG_INDEX_DIR`
  (o `RAG_FAISS_INDEX_PATH` para solo el indice) a una carpeta fuera de esa
  sincronizacion, por ejemplo `C:\venvs\rag_medicina_funcional_data\index`.
- **`app.py` se queda colgado sin error** (CPU casi en cero, no avanza) si
  `gradio` se importa antes que `sentence-transformers`/PyTorch: el import de
  PyTorch se cuelga indefinidamente. Reproducido de forma consistente en este
  entorno. El fix ya esta aplicado en `app.py` (los imports de `src.*` van
  antes que `import gradio`) - si se reordenan, vuelve a colgarse.

## Nota

Este es un proyecto de aprendizaje sobre la arquitectura de RAG. El contenido de
`data/raw/` proviene de articulos de divulgacion publica sobre medicina funcional
(ver cabecera de cada archivo para la fuente); no constituye asesoria medica.