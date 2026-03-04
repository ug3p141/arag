# HyProTwin A-RAG — Implementation Summary

## Overview

Adapted the A-RAG framework (agentic RAG with hierarchical retrieval tools) for German waterway inspection reports (WSV/HyProTwin). The core insight: 69% of evaluation questions expect answers from a single authoritative document, so the agent uses a "identify then deep-read" strategy instead of vanilla chunk-based RAG.

- **53 tests passing**
- **15 implementation tasks completed**
- **99 documents preprocessed** into 4645 chunks
- **106 evaluation questions** converted from the WSV eval catalog

## Components

| Component | Files |
|-----------|-------|
| LLM client (litellm/Vertex AI) | `src/arag/core/llm.py` |
| Metadata extraction | `src/arag/preprocessing/metadata.py` |
| Structure-aware chunking | `src/arag/preprocessing/chunker.py` |
| read_document tool | `src/arag/tools/read_document.py` |
| metadata_filter tool | `src/arag/tools/metadata_filter.py` |
| semantic_search tool (Qdrant) | `src/arag/tools/semantic_search_qdrant.py` |
| keyword_search tool | `src/arag/tools/keyword_search.py` (existing) |
| Preprocessing script | `scripts/preprocess_wsv.py` |
| Batch runner | `scripts/batch_runner_hyprotwin.py` |
| Qdrant upload | `scripts/upload_to_qdrant.py` |
| Evaluation | `scripts/eval_hyprotwin.py` |
| Eval catalog conversion | `scripts/convert_eval_catalog.py` |
| Config | `configs/hyprotwin.yaml` |
| System prompt | `src/arag/agent/prompts/hyprotwin.txt` |

## Tool Architecture

The agent has 4 tools available in a ReAct loop:

1. **metadata_filter** — Filter document registry by structure name, structure ID, inspection year, or report type. Returns document-level results without embeddings.
2. **semantic_search** — Vector similarity search over chunks via Qdrant. Returns chunk previews with document IDs.
3. **keyword_search** — Lexical search over `chunks.json` for exact term matching.
4. **read_document** — Read the full text of a document by its ID. Used after identifying relevant documents via the other tools.

Typical agent flow: `metadata_filter` → identify document → `read_document` → answer.

## Design Documents

- Design: `docs/plans/2026-03-03-hyprotwin-arag-design.md`
- Implementation plan: `docs/plans/2026-03-03-hyprotwin-implementation.md`

## Quick Start Scripts

### Prerequisites

```bash
# Install dependencies
uv sync --all-extras

# Run tests to verify installation
uv run pytest tests/ -v
```

### Step 1: Preprocess documents

Already done — outputs are committed:
- `data/wsv/chunks.json` (4645 chunks)
- `data/wsv/document_registry.json` (99 documents)

To re-run if documents change:

```bash
uv run python scripts/preprocess_wsv.py --data-dir data/wsv --output-dir data/wsv
```

### Step 2: Convert evaluation catalog

Already done — output is committed:
- `data/wsv/questions.json` (106 questions)

To re-run if the eval CSV changes:

```bash
uv run python scripts/convert_eval_catalog.py \
    --input evalwsv/rag_eval_results_poc.csv \
    --output data/wsv/questions.json
```

### Step 3: Start Qdrant

```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

### Step 4: Upload chunks to Qdrant

```bash
uv run python scripts/upload_to_qdrant.py \
    --chunks data/wsv/chunks.json \
    --collection hyprotwin \
    --config configs/hyprotwin.yaml
```

### Step 5: Run batch evaluation

```bash
uv run python scripts/batch_runner_hyprotwin.py \
    --config configs/hyprotwin.yaml \
    --questions data/wsv/questions.json \
    --output results/hyprotwin/predictions.jsonl
```

The runner supports resume — if interrupted, re-run the same command and it will skip already-completed questions.

### Step 6: Evaluate results

```bash
uv run python scripts/eval_hyprotwin.py \
    --predictions results/hyprotwin/predictions.jsonl
```

This outputs per-category metrics (Faktisch, Querschnittlich, Multi-Hop, Zeitlich) and overall document identification rate.

## Known Limitations

- **wehrfeld filter** not yet implemented — the agent cannot filter by weir field (links/mitte/rechts) via metadata; it must read the full document to distinguish.
- **metadata_filter has no limit parameter** — all matching documents are returned. Fine for 99 docs, should add a limit before scaling to 2000.
- **Qdrant semantic search is optional** — if Qdrant is unavailable, the batch runner falls back to keyword_search + metadata_filter + read_document only.
