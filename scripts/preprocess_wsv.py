#!/usr/bin/env python3
"""Preprocess WSV inspection reports into A-RAG format.

Usage:
    uv run python scripts/preprocess_wsv.py --data-dir data/wsv --output-dir data/wsv
"""

import argparse
import json
from pathlib import Path

from arag.preprocessing.metadata import extract_metadata_from_filename
from arag.preprocessing.chunker import chunk_document, extract_inspection_year


def find_complete_md_files(data_dir: Path) -> list[tuple[Path, str]]:
    """Find all _complete.md files and their parent directory names."""
    results = []
    for structure_dir in sorted(data_dir.iterdir()):
        if not structure_dir.is_dir():
            continue
        for md_file in sorted(structure_dir.glob("*_complete.md")):
            results.append((md_file, structure_dir.name))
    return results


def build_document_id(meta: dict) -> str:
    """Build a human-readable document ID from metadata."""
    if meta.get("doc_type") == "bericht" and meta.get("structure_id"):
        year = meta.get("inspection_year") or (meta["report_date"][:4] if meta.get("report_date") else "unknown")
        report_type = meta.get("report_type", "UNK")
        return f"{meta['structure_id']}_{year}_{report_type}"
    if meta.get("doc_type") == "besichtigungsbericht":
        date = meta.get("report_date", "unknown")
        obj_id = meta.get("object_id", "unknown")
        return f"besichtigung_{obj_id}_{date}"
    if meta.get("doc_type") == "archive":
        return meta.get("archive_code", meta["filename"])
    # Fallback: use filename stem
    return Path(meta["filename"]).stem.replace("_complete", "")


def main():
    parser = argparse.ArgumentParser(description="Preprocess WSV documents for A-RAG")
    parser.add_argument("--data-dir", default="data/wsv", help="WSV data directory")
    parser.add_argument("--output-dir", default="data/wsv", help="Output directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = find_complete_md_files(data_dir)
    print(f"Found {len(files)} _complete.md files")

    all_chunks = []
    registry = {}

    for md_file, dir_name in files:
        text = md_file.read_text(encoding="utf-8")
        meta = extract_metadata_from_filename(md_file.name, directory_name=dir_name)

        # Extract inspection year from content
        inspection_year = extract_inspection_year(text, meta.get("report_date"))
        if inspection_year:
            meta["inspection_year"] = inspection_year

        doc_id = build_document_id(meta)

        # Handle duplicate doc_ids by appending a suffix
        if doc_id in registry:
            suffix = 2
            while f"{doc_id}_{suffix}" in registry:
                suffix += 1
            doc_id = f"{doc_id}_{suffix}"

        chunks = chunk_document(text=text, document_id=doc_id, metadata=meta)
        all_chunks.extend(chunks)

        registry[doc_id] = {
            "source_file": str(md_file),
            "structure_name": meta.get("structure_name"),
            "structure_id": meta.get("structure_id"),
            "inspection_year": meta.get("inspection_year"),
            "report_type": meta.get("report_type"),
            "report_date": meta.get("report_date"),
            "doc_type": meta.get("doc_type"),
            "chunk_ids": [c["id"] for c in chunks],
        }

    # Write outputs
    chunks_path = output_dir / "chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(all_chunks)} chunks to {chunks_path}")

    registry_path = output_dir / "document_registry.json"
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(registry)} documents to {registry_path}")


if __name__ == "__main__":
    main()
