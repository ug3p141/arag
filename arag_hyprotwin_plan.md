# Implementation Plan: A-RAG Adaptation for HyProTwin

**Project:** HyProTwin — Agentic RAG for Waterway Structure Inspection Reports
**Date:** 2026-03-03
**Status:** Draft / Rough Plan
**Reference:** Du et al., "A-RAG: Scaling Agentic Retrieval-Augmented Generation via Hierarchical Retrieval Interfaces" (arXiv:2602.03442), GitHub: Ayanami0730/arag

---

## 1 Motivation and Architectural Fit

The current vanilla vector RAG system suffers from a fundamental architecture mismatch: it treats answer synthesis as a similarity-search-across-corpus task, while 69% of evaluation questions expect answers from a single authoritative document. A-RAG addresses this by giving the LLM agent autonomy over retrieval decisions through a ReAct loop with hierarchical retrieval tools. Instead of hard-coding a fixed retrieve-then-generate pipeline, the agent dynamically chooses between keyword search, semantic search, and deep chunk reading — adapting its strategy per query.

The key properties that make A-RAG suitable for HyProTwin:

- **Autonomous strategy selection** resolves the metadata-routing problem: the agent decides when to use exact keyword matching (structure IDs, years) vs. semantic search (conceptual damage queries).
- **Iterative execution** enables the two-stage "identify document, then deep-read" pattern as emergent agent behaviour rather than hard-coded logic.
- **Interleaved tool use** naturally handles multi-hop and temporal queries via decomposition within the ReAct loop.
- **Context tracker** prevents redundant chunk reads, directly addressing the current noise problem (precision@k = 6.3% despite recall@k = 80%).

---

## 2 Preprocessing Pipeline: Structure-Aware Chunking

### 2.1 Current State

Source documents exist as markdown files (`*_complete.md`) converted from PDF inspection reports. The current pipeline applies fixed-size text chunking with rudimentary semantic boundary detection, then embeds with OpenAI ada-002 (1536 dimensions) into Qdrant. Filenames encode metadata — e.g., `Bericht_6120_SBL_TR_18.02.2010_complete.md` encodes structure ID `6120`, report type `SBL_TR`, and date `18.02.2010` — but this information is not extracted into structured Qdrant payload fields.

### 2.2 Metadata Extraction from Filenames

Build a parser that extracts structured fields from each document filename and (where available) from the document header section. Target fields:

| Field | Source | Example |
|---|---|---|
| `structure_id` | Filename regex | `6120` |
| `structure_name` | Lookup table `structure_id -> name`, or from doc header | `Wehranlage Marktbreit` |
| `report_type` | Filename segment after structure_id | `SBL_TR`, `SBL2_GM`, `SBL2_SR` |
| `report_date` | Filename date pattern `dd.mm.yyyy` | `2010-02-18` |
| `inspection_year` | Derived from report_date or document content | `2009` |
| `structure_type` | Lookup or document header | `Wehranlage`, `Schleuse`, `Bruecke` |
| `wehrfeld` | Document content extraction | `links`, `mitte`, `rechts` |
| `source_filename` | Original filename | `Bericht_6120_SBL_TR_18.02.2010.pdf` |

Implementation notes:

- The `inspection_year` requires care: a report dated 18.02.2010 may cover the 2009 inspection. Parse from document header text where possible (look for "Inspektionsjahr" or "Pruefjahr"), fall back to `report_date.year - 1` as heuristic with manual correction list.
- Maintain a `structure_id -> structure_name` mapping table (likely < 50 entries). Populate initially from Plusmeta metadata if available, extend manually.
- Some filenames follow different patterns (e.g., `414-0017398-0000.pdf`, `612-0001963-0001.pdf`, `95100_BAW_JHWSVPR_Gesamt_Bericht.pdf`). These require a second regex pattern set or manual mapping.

### 2.3 Structure-Aware Chunking

Inspection reports follow a predictable document structure. Replace the fixed-size chunker with a section-aware chunker that operates on the markdown source:

**Step 1 — Section detection.** Parse markdown headings and known structural patterns to identify logical sections:

- **Kopfdaten / Header:** Structure name, ID, inspection date, Pruefer, report type. Typically first 20-50 lines.
- **Bauteil sections:** Each Bauteil (structural component) forms a section. Identify by heading patterns like `### Bauteil: Walze` or numbered component lists.
- **Schaden entries:** Individual damage records within a Bauteil section. Each entry typically contains: Schadenkennung (ID), Schadensklasse (severity 0-4), description, location, photo references. These are the atomic units that must not be split across chunks.
- **Damage summary tables:** Tabular overviews — keep as single chunks.
- **Instandsetzungsmassnahmen:** Recommended repairs section.
- **Gesamtnote / Pruefnote:** Overall rating section.

**Step 2 — Chunking rules:**

- A Schaden entry is the atomic unit. Never split a single damage record across chunks.
- If a Bauteil section with all its Schaden entries fits within the chunk size limit (target: ~1500 tokens), keep it as one chunk.
- If a Bauteil section exceeds the limit, split at Schaden entry boundaries.
- The Kopfdaten section is always a separate chunk (it is the document's "identity card").
- Summary tables and Gesamtnote are separate chunks.

**Step 3 — Contextual header prepending.** Each chunk receives a context prefix derived from its parent hierarchy:

```
[Wehranlage Marktbreit | ID:6120 | Inspektion 2009 | Bericht: SBL_TR | Bauteil: Walze]
```

This prefix is included in both the stored text (for keyword search) and the text sent to the embedding model (for semantic search). It ensures that even in isolation, every chunk carries its provenance. This follows the "contextual chunking" pattern (Anthropic, 2024).

**Step 4 — Output format.** Produce two artefacts per document:

1. **A-RAG chunks file** (`chunks.json`): flat list in A-RAG's expected format:
   ```json
   [
     {"id": "6120_2009_SBL_TR_00", "text": "[Wehranlage Marktbreit | ID:6120 | ...] Kopfdaten: ..."},
     {"id": "6120_2009_SBL_TR_01", "text": "[Wehranlage Marktbreit | ID:6120 | ... | Bauteil: Walze] Schaden [5]: ..."}
   ]
   ```
   The chunk ID encodes `{structure_id}_{inspection_year}_{report_type}_{seq}` for human readability and for use in `chunk_read` adjacency lookups.

2. **Qdrant payloads**: same chunks uploaded to Qdrant with structured metadata fields in the payload for the `metadata_filter` tool (see section 3).

### 2.4 Chunk-to-Document Mapping

Maintain a `document_registry.json` that maps `document_id -> [chunk_ids]`. This enables the agent to, after identifying a relevant document via any search tool, retrieve all chunks belonging to that document via `chunk_read` — implementing the whole-document retrieval strategy without a separate mechanism.

```json
{
  "6120_2009_SBL_TR": {
    "source_file": "Bericht_6120_SBL_TR_18.02.2010_complete.md",
    "structure_id": "6120",
    "structure_name": "Wehranlage Marktbreit",
    "inspection_year": 2009,
    "chunk_ids": ["6120_2009_SBL_TR_00", "6120_2009_SBL_TR_01", "6120_2009_SBL_TR_12"]
  }
}
```

---

## 3 Custom Tool: `metadata_filter`

Add a fourth retrieval tool to A-RAG's `ToolRegistry` that queries Qdrant's structured filtering API, bypassing embedding similarity entirely. This addresses the core finding that most questions contain exact identifiers that dense embeddings handle poorly.

### 3.1 Tool Specification

```python
class MetadataFilterTool:
    name = "metadata_filter"
    description = (
        "Filter inspection report chunks by structured metadata. "
        "Use when the question mentions a specific structure name, "
        "structure ID, inspection year, report type, or Wehrfeld. "
        "Returns matching chunks with their metadata and text preview. "
        "Parameters: structure_name (str, optional), structure_id (str, optional), "
        "inspection_year (int, optional), report_type (str, optional), "
        "wehrfeld (str, optional: 'links'|'mitte'|'rechts'), limit (int, default 20)."
    )

    def __init__(self, qdrant_client, collection_name="hyprotwin"):
        self.client = qdrant_client
        self.collection = collection_name

    def execute(self, **kwargs) -> list[dict]:
        conditions = []
        if v := kwargs.get("structure_name"):
            conditions.append(FieldCondition(key="structure_name", match=MatchText(text=v)))
        if v := kwargs.get("structure_id"):
            conditions.append(FieldCondition(key="structure_id", match=MatchValue(value=v)))
        if v := kwargs.get("inspection_year"):
            conditions.append(FieldCondition(key="inspection_year", match=MatchValue(value=v)))
        if v := kwargs.get("report_type"):
            conditions.append(FieldCondition(key="report_type", match=MatchValue(value=v)))
        if v := kwargs.get("wehrfeld"):
            conditions.append(FieldCondition(key="wehrfeld", match=MatchValue(value=v)))

        if not conditions:
            return [{"error": "At least one filter parameter required"}]

        results, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=Filter(must=conditions),
            limit=kwargs.get("limit", 20),
            with_payload=True
        )
        return [{
            "chunk_id": r.id,
            "text_preview": r.payload["text"][:300],
            "structure_name": r.payload.get("structure_name"),
            "inspection_year": r.payload.get("inspection_year"),
            "report_type": r.payload.get("report_type"),
            "document_id": r.payload.get("document_id")
        } for r in results]
```

### 3.2 Registration

Register alongside the three built-in tools. The agent learns to use it from the tool description in the system prompt — no workflow modification needed.

---

## 4 Retrieval Backend: Qdrant Integration (Option A)

### 4.1 Approach

Replace the internals of A-RAG's `SemanticSearchTool` to query the existing Qdrant collection instead of the local FAISS index. The tool's interface (input parameters, output format) stays unchanged so the agent prompt and ReAct loop require no modification.

### 4.2 Implementation Sketch

```python
class SemanticSearchTool:
    name = "semantic_search"
    description = (
        "Semantic search over inspection report chunks using dense embeddings. "
        "Use for conceptual queries where exact keywords are unknown. "
        "Returns top-k most semantically similar chunks."
    )

    def __init__(self, qdrant_client, collection_name, embedding_fn):
        self.client = qdrant_client
        self.collection = collection_name
        self.embed = embedding_fn

    def execute(self, query: str, top_k: int = 10) -> list[dict]:
        query_vector = self.embed(query)
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )
        return [{
            "chunk_id": r.id,
            "score": r.score,
            "text_preview": r.payload["text"][:300],
            "document_id": r.payload.get("document_id")
        } for r in results]
```

### 4.3 Embedding Function

The `embedding_fn` abstracts the embedding model. For continuity with the existing system, use OpenAI ada-002 initially. This can later be swapped to a multilingual model (e.g., `intfloat/multilingual-e5-large`) without changing the tool interface.

### 4.4 `keyword_search` — Keep As-Is

A-RAG's built-in `keyword_search` operates on raw chunk text via case-insensitive lexical matching with scoring by keyword occurrence x length. This works directly on chunks.json and needs no Qdrant integration. The contextual headers prepended during chunking (section 2.3) ensure that structure names and IDs are present in every chunk's text.

### 4.5 `chunk_read` — Minor Adaptation

A-RAG's `chunk_read` retrieves full chunk content by ID and supports +/-1 adjacency for context. The chunk ID scheme from section 2.4 (`{structure_id}_{year}_{type}_{seq}`) makes adjacency meaningful: `chunk_read("6120_2009_SBL_TR_05")` with context will also return chunks `_04` and `_06` from the same document.

Consider extending the adjacency window to +/-2 for longer damage table sections.

---

## 5 LLM Backend: Anthropic API Adaptation

### 5.1 Motivation

Replace OpenAI-compatible backend with Anthropic API to use Claude Sonnet 4.5 or Opus 4.6 as the agent LLM. These models have 200k token context windows, eliminating token budget concerns. Their strong tool-use and reasoning capabilities are directly what A-RAG is designed to exploit.

### 5.2 Modify `src/arag/core/llm.py`

A-RAG's `LLMClient` currently wraps an OpenAI-compatible chat completions API. Replace with Anthropic's messages API:

```python
import anthropic

class LLMClient:
    def __init__(self, model="claude-sonnet-4-5-20250929", api_key=None):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.total_cost = 0.0

    def chat(self, messages: list, tools: list = None,
             max_tokens: int = 16384, temperature: float = 0.0) -> dict:
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        response = self.client.messages.create(**kwargs)
        self._track_cost(response.usage)
        return self._parse_response(response)

    def _convert_tools(self, tools: list) -> list:
        """Convert A-RAG tool definitions to Anthropic tool format."""
        return [{
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["parameters"]
        } for t in tools]

    def _parse_response(self, response) -> dict:
        """Map Anthropic response format to A-RAG's expected format."""
        result = {"content": None, "tool_calls": []}
        for block in response.content:
            if block.type == "text":
                result["content"] = block.text
            elif block.type == "tool_use":
                result["tool_calls"].append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input
                })
        return result
```

### 5.3 ReAct Loop Compatibility

A-RAG's `BaseAgent` implements the ReAct loop by: (1) sending conversation with tool definitions to LLM, (2) parsing tool calls, (3) executing tools and appending results, (4) repeating until final answer or max_loops. This loop is LLM-agnostic as long as `LLMClient` returns a consistent format. The adapter above handles this. Anthropic's native tool-use support maps cleanly to A-RAG's pattern.

### 5.4 Configuration

```yaml
# configs/hyprotwin.yaml
llm:
  provider: "anthropic"
  model: "claude-sonnet-4-5-20250929"
  temperature: 0.0
  max_tokens: 16384

agent:
  max_loops: 15
  max_token_budget: 200000
  verbose: true

embedding:
  provider: "openai"
  model: "text-embedding-ada-002"

qdrant:
  host: "localhost"
  port: 6333
  collection: "hyprotwin"

data:
  chunks_file: "data/hyprotwin/chunks.json"
  document_registry: "data/hyprotwin/document_registry.json"
```

---

## 6 System Prompt Customization

Replace the generic A-RAG system prompt in `src/arag/agent/prompts/` with:

```
You are an expert assistant for German waterway structure inspection reports
(Bauwerksinspektionen nach DIN 1076 / VV-WSV 2101). The corpus contains
approximately 2000 inspection reports (Berichte) for hydraulic structures
including Wehranlagen, Schleusen, and Bruecken along German federal waterways.

DOMAIN KNOWLEDGE:
- Each report documents one inspection of one structure at one point in time.
- Structures have numeric IDs (e.g. "6120") and names (e.g. "Wehranlage Marktbreit").
- Wehranlagen may have multiple Wehrfelder (links, mitte, rechts), each inspected separately.
- Damage entries are classified by Schadensklasse (severity 0-4, where 4 is most severe).
- Each damage is associated with a Bauteil (structural component, e.g. Walze,
  Sohldichtung, Stahlsegment).
- Reports contain: Kopfdaten (metadata), Bauteil sections with Schaden entries,
  damage tables, Pruefnote/Gesamtnote (overall rating), and
  Instandsetzungsmassnahmen (recommended repairs).
- Report filenames encode metadata: Bericht_{structure_id}_{report_type}_{date}.pdf

RETRIEVAL STRATEGY GUIDANCE:
- For questions mentioning a specific structure and/or year: START with
  metadata_filter to narrow to the correct report(s), then use chunk_read
  to read relevant sections in detail.
- For questions about specific damage types or Bauteile across the corpus:
  use semantic_search, then refine with keyword_search for specific terms.
- For cross-structure or temporal comparison questions: decompose into
  sub-queries per inspection year, gather evidence via metadata_filter +
  chunk_read for each year, then synthesize.
- When you find a relevant chunk, read adjacent chunks (+/-1 or +/-2) to get
  full context of the Bauteil section or damage table.
- Prefer reading entire relevant sections over collecting isolated snippets.

ANSWER GUIDELINES:
- Always cite the specific report (by structure name, ID, and inspection year).
- When reporting Schadensklasse, include the associated Bauteil and Schadenkennung.
- If the corpus does not contain sufficient information, state this explicitly.
- Respond in the same language as the question (typically German).
```

---

## 7 Evaluation Catalog Reformatting

### 7.1 Current Format

The BAW evaluation catalog is a semicolon-separated CSV (`rag_eval_results_poc.csv`) with columns: id, kategorie, question, expected_docs, expected_docs_count, retrieved_sources, retrieved_basenames, matched_docs, matched_count, retrieval_score, hit_rate, precision_at_k, recall_at_k, f1_at_k, ndcg_at_k, ground_truth, answer, error.

Relevant columns for A-RAG evaluation: `id`, `question`, `expected_docs`, `ground_truth`.

### 7.2 A-RAG Expected Format

```json
[
  {"id": "q001", "question": "...", "answer": "..."},
  ...
]
```

### 7.3 Conversion Script

```python
# scripts/convert_baw_catalog.py
import pandas as pd
import json

df = pd.read_csv("rag_eval_results_poc.csv", sep=";", encoding="utf-8-sig")

questions = []
for _, row in df.iterrows():
    questions.append({
        "id": f"q{row['id']:03d}",
        "question": row["question"],
        "answer": row["ground_truth"],
        "metadata": {
            "kategorie": row["kategorie"],
            "expected_docs": row["expected_docs"],
            "expected_docs_count": int(row["expected_docs_count"])
        }
    })

with open("data/hyprotwin/questions.json", "w", encoding="utf-8") as f:
    json.dump(questions, f, ensure_ascii=False, indent=2)

print(f"Converted {len(questions)} questions")
```

### 7.4 Extended Evaluation

A-RAG's `eval.py` uses LLM-as-judge and contain-match accuracy. Extend to also measure:

- **Document identification accuracy**: Does the agent's retrieval trajectory include chunks from `expected_docs`? Parse the agent's tool call log.
- **Per-category breakdown**: Report accuracy separately for Faktisch, Querschnittlich, Multi-Hop, Zeitlich.
- **Retrieval efficiency**: Count tool calls and tokens consumed per question (A-RAG's context tracker already logs this).

---

## 8 Implementation Phases

### Phase 1: Infrastructure (Est. 2-3 days)

- [ ] Set up A-RAG repository fork/clone
- [ ] Implement Anthropic API adapter in `llm.py` (section 5)
- [ ] Verify basic ReAct loop works with Claude + built-in tools on a toy example
- [ ] Convert evaluation catalog (section 7)

### Phase 2: Preprocessing (Est. 3-5 days)

- [ ] Build filename metadata parser (section 2.2)
- [ ] Implement structure-aware chunker for markdown reports (section 2.3)
- [ ] Generate contextual headers and chunk IDs (sections 2.3-2.4)
- [ ] Produce `chunks.json` and `document_registry.json`
- [ ] Upload chunks with structured payloads to Qdrant

### Phase 3: Tool Integration (Est. 2-3 days)

- [ ] Implement `MetadataFilterTool` connecting to Qdrant (section 3)
- [ ] Adapt `SemanticSearchTool` to query Qdrant (section 4.2)
- [ ] Verify `KeywordSearchTool` works with new chunks.json
- [ ] Adapt `ReadChunkTool` for new chunk ID scheme and adjacency
- [ ] Register all four tools, write domain system prompt (section 6)

### Phase 4: Evaluation (Est. 2-3 days)

- [ ] Run full 106-question evaluation with A-RAG
- [ ] Compare against vanilla RAG baseline on same questions
- [ ] Analyse per-category accuracy, document identification rate, tool call patterns
- [ ] Identify failure modes, iterate on system prompt and tool descriptions

### Phase 5: Iteration (Ongoing)

- [ ] Tune chunk sizes and adjacency window based on failure analysis
- [ ] Experiment with alternative embedding models for German text
- [ ] Test with Claude Opus 4.6 for complex multi-hop questions
- [ ] Consider adding a `document_summary_search` tool for broad Querschnittlich questions

---

## 9 Risks and Open Questions

| Risk | Mitigation |
|---|---|
| Inspection year extraction from filenames is ambiguous (report date != inspection year) | Build manual correction table; validate against doc header text; flag uncertain cases |
| Section detection in markdown may be unreliable across heterogeneous report formats | Start with most common format, extend iteratively; fall back to fixed-size for unrecognised formats |
| Anthropic API cost at scale (106 questions x ~15 loops x ~4k tokens/loop) | Estimate: ~6M tokens per eval run = $20-40 with Sonnet; use Haiku for rapid iteration |
| Agent may over-iterate on simple Faktisch questions | Tune max_loops; domain prompt guides quick resolution when metadata_filter returns clear match |
| A-RAG codebase is young (3 commits, Feb 2026) | Core is ~500 lines, easy to read and patch. MIT license allows full modification. |
