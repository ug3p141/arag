#!/usr/bin/env python3
"""Convert WSV evaluation CSV to A-RAG questions.json format.

Usage:
    uv run python scripts/convert_eval_catalog.py \
        --input evalwsv/rag_eval_results_poc.csv \
        --output data/wsv/questions.json
"""

import argparse
import csv
import json


def main():
    parser = argparse.ArgumentParser(description="Convert WSV eval CSV to A-RAG format")
    parser.add_argument("--input", default="evalwsv/rag_eval_results_poc.csv")
    parser.add_argument("--output", default="data/wsv/questions.json")
    args = parser.parse_args()

    questions = []
    with open(args.input, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            questions.append({
                "id": f"q{int(row['id']):03d}",
                "question": row["question"],
                "answer": row["ground_truth"],
                "metadata": {
                    "kategorie": row["kategorie"],
                    "expected_docs": row["expected_docs"],
                    "expected_docs_count": int(row["expected_docs_count"]),
                    "vanilla_answer": row["answer"],
                    "vanilla_retrieval_score": (
                        float(row["retrieval_score"]) if row.get("retrieval_score") else None
                    ),
                    "vanilla_precision_at_k": (
                        float(row["precision_at_k"]) if row.get("precision_at_k") else None
                    ),
                },
            })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(questions)} questions to {args.output}")

    from collections import Counter

    cats = Counter(q["metadata"]["kategorie"] for q in questions)
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
