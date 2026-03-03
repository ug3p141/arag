# Design: A-RAG Adaptation for HyProTwin WSV Inspection Reports

**Date:** 2026-03-03
**Status:** Approved
**Approach:** B — Adapted A-RAG with full toolset and whole-document reading

---

## Problem Statement

A research group is building a RAG system over ~2000 German waterway inspection reports. Their vanilla vector RAG (chunk + embed + retrieve) achieves only 5.2% precision@k and ~20% hit rate on a 106-question evaluation set. The core mismatch: 69% of questions expect answers from a single authoritative document, but the vanilla approach scatters retrieval across the entire corpus. Answers require deep reading of structured documents, not similarity-based snippet extraction.

A-RAG (Du et al., 2026) addresses this by giving an LLM agent autonomy over retrieval strategy via a ReAct loop with hierarchical tools.

## Scope

- Validate on a 99-document subset (2 structures: Wehranlage Marktbreit, Schiffsschleusenanlage Altenrheine)
- Beat the vanilla RAG baseline on the same 106 evaluation questions
- Scale to full ~2000-document corpus later

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Vector store | Qdrant | Team already uses it |
| LLM backend | Vertex AI via litellm | Team infrastructure; consistent with HyproTwin |
| LLM model | Claude Sonnet 4.5 (default) | Cost/capability balance; switchable via config |
| Image descriptions | Excluded for now | Text-only; image descriptions available for later enhancement |
| Reading strategy | Whole-document read | Matches "answer in one document" pattern; docs are 7-39KB |

---

## Architecture

```
+---------------------------------------------+
|  Evaluation Driver (batch_runner.py)         |
|  106 questions -> agent -> predictions.jsonl |
+---------------------------------------------+
|  Agent (BaseAgent + ReAct loop)              |
|  Claude Sonnet 4.5 via Vertex AI / litellm   |
|  Domain system prompt (German WSV inspector) |
+---------------------------------------------+
|  Tool Registry (4 tools)                     |
|  metadata_filter | semantic_search           |
|  keyword_search  | read_document             |
+---------------------------------------------+
|  Data Layer                                  |
|  Qdrant (vectors + metadata payloads)        |
|  chunks.json (keyword search)                |
|  document_registry.json (doc -> file mapping)|
|  _complete.md files (full document text)     |
+---------------------------------------------+
```

### Agent Flow Examples

**Faktisch question** ("Welche Schaeden mit hoechster Schadensklasse im Pruefbericht 2009 an Wehranlage Marktbreit, linkes Wehrfeld?"):

1. `metadata_filter(structure_name="Marktbreit", inspection_year=2009, wehrfeld="links")` -> document IDs
2. `read_document(document_id="6120_2009_SBL_TR")` -> full markdown
3. Agent reads, identifies Schadensklasse 3 entries, formulates answer
4. Done in ~3 tool calls, ~1 loop

**Zeitlich question** ("Wie hat sich der Schaden an der Seitendichtung zwischen 2018 und 2021 veraendert?"):

1. `metadata_filter(structure_name="Marktbreit", inspection_year=2018)` -> doc IDs
2. `metadata_filter(structure_name="Marktbreit", inspection_year=2021)` -> doc IDs
3. `read_document(...)` for each relevant report
4. Agent compares damage descriptions across years
5. Done in ~4-6 tool calls, ~2-3 loops

---

## Preprocessing Pipeline

**Input:** 99 `_complete.md` files across 2 structure directories in `data/wsv/`

**Output:**
- `data/wsv/chunks.json` — chunked text for keyword search and semantic indexing
- `data/wsv/document_registry.json` — document_id -> metadata + chunk list + source file path
- Qdrant collection `hyprotwin` with chunks + structured metadata payloads

### Metadata Extraction

Parse from directory name and filename:

| Field | Source | Example |
|-------|--------|---------|
| `structure_id` | Filename regex | `6120` |
| `structure_name` | Directory name + lookup table | `Wehranlage Marktbreit` |
| `report_type` | Filename segment | `SBL_TR`, `SBL2_GM`, `SBL2_SR` |
| `report_date` | Filename date pattern | `2010-02-18` |
| `inspection_year` | Document content or report_date heuristic | `2009` |
| `wehrfeld` | Document content extraction | `links`, `mitte`, `rechts` |

Notes:
- Directory-level structure_id (`213-6326002`) differs from report-level (`6120`). Maintain a manual lookup table (<10 entries for this subset).
- Inspection year: parse from document header ("Inspektionsjahr", "Pruefjahr"), fall back to `report_date.year` with manual corrections.
- Some filenames follow non-standard patterns (e.g. `414-0017398-0000.pdf`). Handle with secondary regex or manual mapping.

### Chunking Strategy

Chunks serve the **discovery** role (semantic/keyword search finds them). The **reading** role is handled by `read_document`.

- **Kopfdaten chunk** (seq 00): document header/identity card (~200 tokens)
- **Bauteil+Schaden chunks**: each Bauteil section with its damage entries. Never split a single Schaden entry. Target ~1000-1500 tokens.
- **Summary/Gesamtnote chunk** (last seq): overall rating section
- **Contextual header** prepended to every chunk: `[Wehranlage Marktbreit | ID:6120 | Inspektion 2009 | SBL_TR | Bauteil: Walze]`

Chunk ID scheme: `{structure_id}_{year}_{report_type}_{seq}` e.g. `6120_2009_SBL_TR_00`

### Document Registry

```json
{
  "6120_2009_SBL_TR": {
    "source_file": "data/wsv/30_.../Bericht_6120_SBL_TR_18.02.2010_complete.md",
    "structure_id": "6120",
    "structure_name": "Wehranlage Marktbreit",
    "inspection_year": 2009,
    "report_type": "SBL_TR",
    "report_date": "2010-02-18",
    "chunk_ids": ["6120_2009_SBL_TR_00", "...", "6120_2009_SBL_TR_12"]
  }
}
```

---

## Tool Specifications

### Tool 1: `metadata_filter`

Queries Qdrant structured filtering — no embeddings. Returns **document-level** results (deduplicated from chunk hits).

```
metadata_filter(
  structure_name: str = None,   # fuzzy text match
  structure_id: str = None,     # exact match
  inspection_year: int = None,  # exact match
  report_type: str = None,      # exact match
  wehrfeld: str = None,         # exact: "links"|"mitte"|"rechts"
  limit: int = 20
)
-> list of {document_id, structure_name, inspection_year, report_type, text_preview}
```

### Tool 2: `semantic_search`

Qdrant vector search on chunk embeddings. For conceptual queries without exact identifiers.

```
semantic_search(
  query: str,
  top_k: int = 10
)
-> list of {chunk_id, document_id, score, text_preview, structure_name, inspection_year}
```

Embedding model: team's existing ada-002, swappable to multilingual model later.

### Tool 3: `keyword_search`

A-RAG's built-in lexical search on chunks.json. Exact term matching. No changes needed beyond loading the new chunks.json.

```
keyword_search(
  query: str,
  top_k: int = 10
)
-> list of {chunk_id, document_id, score, text_preview}
```

### Tool 4: `read_document` (new)

Primary reading tool. Returns full `_complete.md` content via document registry lookup.

```
read_document(
  document_id: str   # e.g. "6120_2009_SBL_TR"
)
-> full document text (7-39KB)
```

---

## LLM Backend: Vertex AI via litellm

Replace A-RAG's raw `requests`-based OpenAI client with `litellm.completion()`, matching the team's existing HyproTwin infrastructure.

```python
import litellm

class LLMClient:
    def __init__(self, model, vertex_project, vertex_location, vertex_credentials):
        self.model = model
        self.vertex_kwargs = {
            "vertex_project": vertex_project,
            "vertex_location": vertex_location,
            "vertex_credentials": vertex_credentials,
        }

    def chat(self, messages, tools=None, **kwargs):
        response = litellm.completion(
            model=self.model,
            messages=messages,
            tools=tools,
            **self.vertex_kwargs,
            **kwargs,
        )
        return self._parse_response(response)
```

Configuration:
```yaml
llm:
  model: "vertex_ai/claude-sonnet-4-5-20250929"
  vertex_project: "hyprotwin"
  vertex_location: "europe-west4"
  vertex_credentials: "vertex_credentials/hyprotwin-58e6370ca989.json"
  temperature: 0.0
  max_tokens: 16384
```

---

## System Prompt

```
You are an expert assistant for German waterway structure inspection reports
(Bauwerksinspektionen nach DIN 1076 / VV-WSV 2101). The corpus contains
inspection reports for hydraulic structures including Wehranlagen, Schleusen,
and Bruecken along German federal waterways.

DOMAIN KNOWLEDGE:
- Each report documents one inspection of one structure at one point in time.
- Structures have numeric IDs (e.g. "6120") and names (e.g. "Wehranlage Marktbreit").
- Wehranlagen may have multiple Wehrfelder (links, mitte, rechts).
- Damage entries are classified by Schadensklasse (severity 0-4, where 4 is most severe).
- Each damage is associated with a Bauteil (structural component).
- Reports contain: Kopfdaten, Bauteil sections with Schaden entries, damage tables,
  Pruefnote/Gesamtnote, and Instandsetzungsmassnahmen.

RETRIEVAL STRATEGY:
1. When the question names a structure and/or year: START with metadata_filter
   to identify the right report(s), then read_document to read them fully.
2. For conceptual queries without specific identifiers: use semantic_search
   to discover relevant chunks, note their document_ids, then read_document.
3. For temporal comparison questions: call metadata_filter for each year,
   read_document for each report, then compare.
4. Use keyword_search for exact terms (Schadenkennung codes, Bauteil names)
   when semantic_search is too broad.
5. ALWAYS read the full document before answering - do not answer from
   chunk previews alone.

ANSWER GUIDELINES:
- Cite the specific report (structure name, ID, inspection year).
- When reporting damages, include Schadensklasse, Bauteil, and Schadenkennung.
- If the corpus lacks sufficient information, say so explicitly.
- Respond in the same language as the question (typically German).
```

---

## Evaluation

### Eval Catalog Conversion

Convert `evalwsv/rag_eval_results_poc.csv` -> `data/wsv/questions.json`:
```json
[
  {
    "id": "q001",
    "question": "...",
    "answer": "...",
    "metadata": {
      "kategorie": "Faktisch",
      "expected_docs": "Bericht_6120_SBL_TR_18.02.2010.pdf",
      "expected_docs_count": 1
    }
  }
]
```

### Metrics

- **LLM-Accuracy**: LLM-as-judge (A-RAG's existing metric)
- **Contain-Match**: string containment check (existing)
- **Document identification rate**: did the agent's trajectory include expected_docs?
- **Per-category breakdown**: Faktisch / Querschnittlich / Multi-Hop / Zeitlich
- **Efficiency**: avg tool calls and loops per question

### Baseline Comparison

The CSV already contains the vanilla RAG's generated answers. Run eval.py on both sets for a direct comparison table.

---

## Cost Estimate

~106 questions x ~3 loops avg x ~5k tokens/loop = ~1.6M tokens per eval run = ~$5-10 with Sonnet via Vertex AI.
