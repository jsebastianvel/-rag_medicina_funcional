"""Harness de evaluacion: mide que tan bien recupera el sistema los
documentos correctos para un set de preguntas con respuesta esperada
conocida (ground truth manual, ya que el corpus es pequeno y curado)."""

import json
import os
from operator import truediv

from src.config import ROOT_DIR, TOP_K_FINAL
from src.retrieval import hybrid_search
from src.generate import generate_answer

QUESTIONS_PATH = os.path.join(ROOT_DIR, "eval", "questions.json")

N_GENERATION_SAMPLES = 3


def load_questions():
    with open(QUESTIONS_PATH, "r", encoding="utf-8-sig") as fh:
        return json.load(fh)


def evaluate_retrieval(questions, k=TOP_K_FINAL):
    per_question = []

    for question in questions:
        query = question["query"]
        expected = set(question["expected_filenames"])

        results = hybrid_search(query, k)
        retrieved_filenames = [item["chunk"]["filename"] for item in results]

        hits = [name for name in retrieved_filenames if name in expected]
        hit_at_k = len(hits) > 0
        precision_at_k = truediv(len(hits), k)

        per_question.append({
            "query": query,
            "expected": sorted(expected),
            "retrieved": retrieved_filenames,
            "hit_at_k": hit_at_k,
            "precision_at_k": precision_at_k,
        })

    n = len(per_question)
    hit_count = sum(1 for item in per_question if item["hit_at_k"])
    hit_rate = truediv(hit_count, n)
    precision_sum = sum(item["precision_at_k"] for item in per_question)
    mean_precision = truediv(precision_sum, n)

    return per_question, hit_rate, mean_precision


def check_generation_faithfulness(questions, n_samples=N_GENERATION_SAMPLES):
    """Corre la generacion completa (con LLM) solo sobre unas pocas preguntas
    de muestra, y verifica que las fuentes citadas coincidan con las
    esperadas. No se corre sobre todo el set para no gastar llamadas de API
    de mas en cada corrida del harness.

    Cada pregunta se evalua en su propio try/except: un error de la API
    (por ejemplo, cuota agotada) en una pregunta no debe tumbar el resto
    del harness."""
    samples = questions[:n_samples]
    results = []

    for question in samples:
        query = question["query"]
        expected = set(question["expected_filenames"])

        try:
            output = generate_answer(query)
        except Exception as exc:
            results.append({
                "query": query,
                "expected": sorted(expected),
                "error": str(exc)[:200],
            })
            continue

        cited_filenames = set(source["filename"] for source in output["sources"])
        faithful = len(cited_filenames.intersection(expected)) > 0

        results.append({
            "query": query,
            "expected": sorted(expected),
            "cited": sorted(cited_filenames),
            "faithful": faithful,
            "answer_preview": output["answer"][:200],
        })

    return results


if __name__ == "__main__":
    questions = load_questions()

    print("=== Evaluacion de retrieval (" + str(len(questions)) + " preguntas, k=" + str(TOP_K_FINAL) + ") ===")
    per_question, hit_rate, mean_precision = evaluate_retrieval(questions)

    for question_result in per_question:
        status = "OK" if question_result["hit_at_k"] else "MISS"
        print("")
        print("[" + status + "] " + question_result["query"])
        print("  esperado=" + str(question_result["expected"]))
        print("  recuperado=" + str(question_result["retrieved"]))
        print("  precision_at_k=" + str(round(question_result["precision_at_k"], 2)))

    print("")
    print("=== Resumen retrieval ===")
    print("Hit rate=" + "{:.1%}".format(hit_rate))
    print("Precision promedio=" + str(round(mean_precision, 3)))

    print("")
    print("=== Fidelidad de la generacion (muestra de " + str(N_GENERATION_SAMPLES) + " preguntas) ===")
    generation_results = check_generation_faithfulness(questions)
    for generation_result in generation_results:
        print("")
        if "error" in generation_result:
            print("[ERROR] " + generation_result["query"])
            print("  " + generation_result["error"])
            continue
        status = "OK" if generation_result["faithful"] else "MISS"
        print("[" + status + "] " + generation_result["query"])
        print("  citado=" + str(generation_result["cited"]))