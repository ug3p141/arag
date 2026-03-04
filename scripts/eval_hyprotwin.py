#!/usr/bin/env python3
"""Extended evaluation for HyProTwin, adding document identification
and per-category metrics on top of A-RAG's standard eval.

Usage:
    uv run python scripts/eval_hyprotwin.py \
        --predictions results/hyprotwin/predictions.jsonl
"""

import argparse
import json
from collections import defaultdict


def check_document_identification(trajectory: list, expected_docs: str) -> bool:
    """Check if the agent's trajectory accessed any of the expected documents."""
    if not expected_docs:
        return True

    expected = set()
    for doc in expected_docs.split(";"):
        doc = doc.strip()
        base = doc.replace(".pdf", "").lower()
        expected.add(base)

    for entry in trajectory:
        tool_name = entry.get("tool_name", "")
        args = entry.get("arguments", {})
        result = entry.get("tool_result", "")

        if tool_name == "read_document":
            doc_id = args.get("document_id", "").lower()
            for exp in expected:
                if exp in doc_id or doc_id in exp:
                    return True

        result_lower = result.lower() if isinstance(result, str) else ""
        for exp in expected:
            if exp in result_lower:
                return True

    return False


def main():
    parser = argparse.ArgumentParser(description="HyProTwin evaluation")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default=None, help="Output summary file")
    args = parser.parse_args()

    predictions = []
    with open(args.predictions, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))

    print(f"Loaded {len(predictions)} predictions")

    category_stats = defaultdict(lambda: {
        "total": 0,
        "doc_identified": 0,
        "errors": 0,
        "total_loops": 0,
        "total_cost": 0.0,
    })

    for pred in predictions:
        meta = pred.get("metadata", {})
        cat = meta.get("kategorie", "Unknown")
        expected_docs = meta.get("expected_docs", "")

        stats = category_stats[cat]
        stats["total"] += 1
        stats["total_loops"] += pred.get("loops", 0)
        stats["total_cost"] += pred.get("total_cost", 0)

        if pred.get("error"):
            stats["errors"] += 1
            continue

        if check_document_identification(pred.get("trajectory", []), expected_docs):
            stats["doc_identified"] += 1

    print("\n" + "=" * 70)
    print("HyProTwin Evaluation Report")
    print("=" * 70)

    total_all = 0
    doc_id_all = 0

    for cat in ["Faktisch", "Querschnittlich", "Multi-Hop", "Zeitlich", "Unknown"]:
        if cat not in category_stats:
            continue
        s = category_stats[cat]
        total_all += s["total"]
        doc_id_all += s["doc_identified"]
        doc_rate = s["doc_identified"] / s["total"] * 100 if s["total"] else 0
        avg_loops = s["total_loops"] / s["total"] if s["total"] else 0
        avg_cost = s["total_cost"] / s["total"] if s["total"] else 0

        print(f"\n{cat} ({s['total']} questions):")
        print(f"  Doc identification: {s['doc_identified']}/{s['total']} ({doc_rate:.1f}%)")
        print(f"  Errors: {s['errors']}")
        print(f"  Avg loops: {avg_loops:.1f}")
        print(f"  Avg cost: ${avg_cost:.4f}")

    overall_rate = doc_id_all / total_all * 100 if total_all else 0
    print(f"\nOverall doc identification: {doc_id_all}/{total_all} ({overall_rate:.1f}%)")

    output_path = args.output or args.predictions.replace(".jsonl", "_eval_summary.json")
    summary = {
        "total_predictions": len(predictions),
        "overall_doc_identification_rate": overall_rate,
        "per_category": {cat: dict(s) for cat, s in category_stats.items()},
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {output_path}")


if __name__ == "__main__":
    main()
