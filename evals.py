"""Batch evaluation script for the StackOne RAG pipeline.

Inspired by the RAGAS evaluation-driven-development pattern:
  https://docs.ragas.io/en/stable/tutorials/rag/

Usage
-----
# Run against the default test dataset (evals/datasets/test_dataset.csv)
python evals.py

# Specify a custom dataset file
python evals.py --dataset evals/datasets/my_dataset.csv

# Override output directory
python evals.py --output-dir evals/experiments

Columns in the dataset CSV
--------------------------
- query         (required) The question to ask the RAG system
- grading_notes (optional) Reference notes used to judge correctness

Metrics reported (all 0-1, higher is better)
--------------------------------------------
- faithfulness      Every claim in the answer is grounded in the retrieved context
- answer_relevancy  The answer actually addresses the question
- context_relevancy The retrieved chunks are relevant to the question
- correctness       Pass/fail: does the answer cover the grading notes?
                    (only computed when grading_notes is present)

Results are written to a timestamped CSV under --output-dir.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import logging
import os
import sys
from pathlib import Path

import chromadb

# ---------------------------------------------------------------------------
# Bootstrap: ensure the project root is on sys.path so we can import
# ai_exercise even when running the script directly (not as a module).
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_exercise.constants import SETTINGS, chroma_client, llm_provider  # noqa: E402
from ai_exercise.evaluation.evaluator import RAGEvaluator  # noqa: E402
from ai_exercise.llm.rag_chat_prompts import RAGChatPrompts  # noqa: E402
from ai_exercise.loading.openapi_spec_loader import build_and_add_documents  # noqa: E402
from ai_exercise.retrieval.vector_store import ChromaVectorStore  # noqa: E402

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

PASS_THRESHOLD = 0.6  # scores above this are considered passing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_dataset(path: str) -> list[dict]:
    """Load a CSV dataset and return a list of row dicts."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        print(f"[warn] Dataset file is empty: {path}")
    return rows


def _get_or_build_vector_store():
    """Return the existing persistent vector store (Chroma), loading docs if empty."""
    vs = ChromaVectorStore(
        chroma_client, llm_provider.embeddings, SETTINGS.collection_name
    )
    # Peek at the collection count — if nothing is loaded yet, load now.
    collection = chroma_client.get_or_create_collection(SETTINGS.collection_name)
    count = collection.count()
    if count == 0:
        print("[info] Vector store is empty — loading OpenAPI specs now (this may take a minute)...")
        chroma_store = vs.reset()
        loaded = build_and_add_documents(chroma_store)
        print(f"[info] Loaded {loaded} document chunks.")
        return chroma_store
    else:
        print(f"[info] Using existing vector store ({count} chunks).")
        return vs.get()


def _score_correctness(llm_provider, query: str, answer: str, grading_notes: str) -> str:
    """Return 'pass' or 'fail' by asking the LLM to compare answer vs grading notes."""
    prompt = (
        "You are a strict grader. Check whether the response covers the key points "
        "from the grading notes.\n\n"
        f"Question: {query}\n\n"
        f"Response: {answer}\n\n"
        f"Grading Notes: {grading_notes}\n\n"
        "Does the response adequately cover the grading notes? "
        "Reply with exactly one word: 'pass' or 'fail'."
    )
    raw = llm_provider.get_completion(prompt).strip().lower()
    return "pass" if "pass" in raw else "fail"


def _bar(score: float, width: int = 10) -> str:
    """Simple ASCII progress bar for a 0-1 score."""
    filled = round(score * width)
    return "[" + "#" * filled + "." * (width - filled) + f"] {score:.2f}"


def _print_results_table(results: list[dict]) -> None:
    """Print a formatted table of evaluation results to stdout."""
    col_widths = {
        "query": 45,
        "faithfulness": 18,
        "answer_relevancy": 20,
        "context_relevancy": 21,
        "correctness": 12,
    }

    header = (
        f"{'Query':<{col_widths['query']}} "
        f"{'Faithfulness':<{col_widths['faithfulness']}} "
        f"{'Ans Relevancy':<{col_widths['answer_relevancy']}} "
        f"{'Ctx Relevancy':<{col_widths['context_relevancy']}} "
        f"{'Correctness'}"
    )
    sep = "-" * len(header)

    print("\n" + sep)
    print(header)
    print(sep)

    for r in results:
        query_short = r["query"][:43] + ".." if len(r["query"]) > 43 else r["query"]
        faith = _bar(r["faithfulness"])
        ans_rel = _bar(r["answer_relevancy"])
        ctx_rel = _bar(r["context_relevancy"])
        correctness = r.get("correctness", "n/a")

        print(
            f"{query_short:<{col_widths['query']}} "
            f"{faith:<{col_widths['faithfulness']}} "
            f"{ans_rel:<{col_widths['answer_relevancy']}} "
            f"{ctx_rel:<{col_widths['context_relevancy']}} "
            f"{correctness}"
        )

    print(sep)

    # Aggregate stats
    avg_faith = sum(r["faithfulness"] for r in results) / len(results)
    avg_ans = sum(r["answer_relevancy"] for r in results) / len(results)
    avg_ctx = sum(r["context_relevancy"] for r in results) / len(results)
    pass_count = sum(1 for r in results if r.get("correctness") == "pass")
    n_graded = sum(1 for r in results if r.get("correctness") in ("pass", "fail"))

    print(
        f"\n{'AVERAGES':<{col_widths['query']}} "
        f"{avg_faith:<{col_widths['faithfulness']}.2f} "
        f"{avg_ans:<{col_widths['answer_relevancy']}.2f} "
        f"{avg_ctx:.2f}"
    )
    if n_graded:
        print(f"Correctness (pass rate): {pass_count}/{n_graded} ({100*pass_count//n_graded}%)")
    print()


def _save_results_csv(results: list[dict], output_dir: str) -> str:
    """Write results to a timestamped CSV and return the file path."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"experiment_{timestamp}.csv")
    fieldnames = [
        "query", "answer", "faithfulness", "answer_relevancy",
        "context_relevancy", "correctness", "retrieved_chunks",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_eval(dataset_path: str, output_dir: str) -> None:
    """Run the full evaluation loop."""
    rows = _load_dataset(dataset_path)
    if not rows:
        print("[error] No test cases found. Exiting.")
        sys.exit(1)

    print(f"\nLoaded {len(rows)} test case(s) from {dataset_path}")

    # Set up the vector store and retriever
    vector_store = _get_or_build_vector_store()
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": SETTINGS.k_neighbors,
            "score_threshold": SETTINGS.score_threshold,
        },
    )
    evaluator = RAGEvaluator(llm_provider)

    results = []
    for i, row in enumerate(rows, 1):
        query = row.get("query", "").strip()
        grading_notes = row.get("grading_notes", "").strip()

        if not query:
            print(f"[warn] Skipping row {i} — missing 'query' field.")
            continue

        print(f"\n[{i}/{len(rows)}] Evaluating: {query[:70]}...")

        # Retrieve + generate
        relevant_docs = retriever.invoke(query)
        relevant_chunks = [doc.page_content for doc in relevant_docs]
        prompt = RAGChatPrompts.answer(query=query, context=relevant_chunks)
        answer = llm_provider.get_completion(prompt)

        print(f"        Answer: {answer[:120]}{'...' if len(answer) > 120 else ''}")
        print(f"        Retrieved {len(relevant_chunks)} chunk(s).")

        # Score with LLM-as-judge
        scores = evaluator.evaluate(query=query, context=relevant_chunks, answer=answer)

        # Optional correctness check against grading notes
        correctness = "n/a"
        if grading_notes:
            correctness = _score_correctness(llm_provider, query, answer, grading_notes)

        result = {
            "query": query,
            "answer": answer,
            "faithfulness": scores["faithfulness"],
            "answer_relevancy": scores["answer_relevancy"],
            "context_relevancy": scores["context_relevancy"],
            "correctness": correctness,
            "retrieved_chunks": " | ".join(c[:100] for c in relevant_chunks),
        }
        results.append(result)

    if not results:
        print("[error] No results produced.")
        sys.exit(1)

    _print_results_table(results)

    out_path = _save_results_csv(results, output_dir)
    print(f"Results saved to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch evaluation runner for the StackOne RAG pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        default="evals/datasets/test_dataset.csv",
        help="Path to the CSV test dataset (default: evals/datasets/test_dataset.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="evals/experiments",
        help="Directory to write result CSVs (default: evals/experiments)",
    )
    args = parser.parse_args()

    # Change cwd to project root so relative paths resolve correctly
    os.chdir(ROOT)

    run_eval(dataset_path=args.dataset, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
