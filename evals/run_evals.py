# evals/run_eval.py
"""
Evaluation harness — uruchamia golden dataset przez pipeline i liczy metryki.

Metryki:
  - answer_has_key_fact     : odpowiedź zawiera kluczowy fakt
  - citation_hit_rate       : oczekiwane dokumenty zostały zacytowane
  - no_answer_precision     : "nie wiem" tylko dla out-of-scope pytań
  - avg_latency_ms          : średnia latencja
  - per_category breakdown  : metryki per typ pytania

Uruchomienie:
  uv run python evals/run_eval.py
  uv run python evals/run_eval.py --output evals/results/run_001.json
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

# Dodaj src do PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import typer
from loguru import logger

from evals.golden_dataset.questions import GOLDEN_DATASET, GoldenQuestion
from tsl_rag.core.models import RetrievalRequest
from tsl_rag.generation.generator import RAGGenerator
from tsl_rag.retrieval.retriever import HybridRetriever

app = typer.Typer(add_completion=False)


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------


async def evaluate_question(
    question: GoldenQuestion,
    retriever: HybridRetriever,
    generator: RAGGenerator,
) -> dict:
    t0 = time.monotonic()

    request = RetrievalRequest(
        query=question.question,
        top_k=20,
        rerank_top_n=5,
    )

    results = await retriever.retrieve(request)
    response = await generator.generate(question.question, results)

    latency_ms = int((time.monotonic() - t0) * 1000)

    # --- Metryki ---
    answer_lower = response.answer.lower()

    # 1. Kluczowy fakt w odpowiedzi (case-insensitive, partial match)
    key_facts = [f.strip() for f in question.expected_answer.lower().split(",")]
    fact_hits = sum(1 for f in key_facts if f in answer_lower)
    answer_score = fact_hits / len(key_facts) if key_facts else 1.0

    # 2. Cytowania dokumentów
    cited_docs = {c.document_id for c in response.citations}
    expected_set = set(question.expected_docs)
    if expected_set:
        citation_hit = len(cited_docs & expected_set) / len(expected_set)
    else:
        citation_hit = 1.0  # out-of-scope: brak cytowań = poprawne

    # 3. "Nie wiem" precision
    is_out_of_scope = question.category == "out_of_scope"
    correctly_refused = not response.has_answer and is_out_of_scope
    incorrectly_refused = not response.has_answer and not is_out_of_scope

    return {
        "question": question.question,
        "category": question.category,
        "expected_docs": question.expected_docs,
        "answer_score": round(answer_score, 3),
        "citation_hit_rate": round(citation_hit, 3),
        "has_answer": response.has_answer,
        "correctly_refused": correctly_refused,
        "incorrectly_refused": incorrectly_refused,
        "latency_ms": latency_ms,
        "cited_docs": list(cited_docs),
        "answer_preview": response.answer[:200],
    }


async def run_evaluation(output_path: Path | None) -> None:
    results_list: list[dict] = []

    async with HybridRetriever() as retriever:
        generator = RAGGenerator()

        for i, question in enumerate(GOLDEN_DATASET):
            logger.info(f"[{i+1}/{len(GOLDEN_DATASET)}] {question.question[:70]}")
            result = await evaluate_question(question, retriever, generator)
            results_list.append(result)
            _print_result(result)

    # --- Agregacja ---
    summary = _aggregate(results_list)
    _print_summary(summary)

    output = {"summary": summary, "results": results_list}

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Results saved to {output_path}")


def _aggregate(results: list[dict]) -> dict:
    n = len(results)
    categories: dict[str, list[dict]] = {}
    for r in results:
        categories.setdefault(r["category"], []).append(r)

    per_cat = {}
    for cat, items in categories.items():
        per_cat[cat] = {
            "count": len(items),
            "avg_answer_score": round(sum(i["answer_score"] for i in items) / len(items), 3),
            "avg_citation_hit": round(sum(i["citation_hit_rate"] for i in items) / len(items), 3),
        }

    out_of_scope = [r for r in results if r["category"] == "out_of_scope"]
    in_scope = [r for r in results if r["category"] != "out_of_scope"]

    return {
        "total_questions": n,
        "avg_answer_score": round(sum(r["answer_score"] for r in results) / n, 3),
        "avg_citation_hit_rate": round(sum(r["citation_hit_rate"] for r in results) / n, 3),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / n),
        "refusal_precision": round(
            sum(1 for r in out_of_scope if r["correctly_refused"]) / len(out_of_scope), 3
        )
        if out_of_scope
        else None,
        "false_refusal_rate": round(
            sum(1 for r in in_scope if r["incorrectly_refused"]) / len(in_scope), 3
        )
        if in_scope
        else None,
        "per_category": per_cat,
    }


def _print_result(r: dict) -> None:
    status = "✓" if r["answer_score"] >= 0.5 and r["citation_hit_rate"] >= 0.5 else "✗"
    print(
        f"  {status} [{r['category']:15s}] "
        f"fact={r['answer_score']:.2f} cite={r['citation_hit_rate']:.2f} "
        f"{r['latency_ms']}ms"
    )


def _print_summary(s: dict) -> None:
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Questions      : {s['total_questions']}")
    print(f"  Answer score   : {s['avg_answer_score']:.3f}  (key facts found)")
    print(f"  Citation hit   : {s['avg_citation_hit_rate']:.3f}  (correct docs cited)")
    print(f"  Refusal prec.  : {s['refusal_precision']}  (out-of-scope refused)")
    print(f"  False refusals : {s['false_refusal_rate']}  (in-scope refused)")
    print(f"  Avg latency    : {s['avg_latency_ms']}ms")
    print("\n  Per category:")
    for cat, v in s["per_category"].items():
        print(
            f"    {cat:20s} n={v['count']} fact={v['avg_answer_score']:.2f} cite={v['avg_citation_hit']:.2f}"
        )
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    output: Path = typer.Option(None, "--output", "-o", help="Zapisz wyniki do JSON"),  # noqa: B008
) -> None:
    """Uruchom eval harness na golden dataset."""
    asyncio.run(run_evaluation(output))


if __name__ == "__main__":
    app()
