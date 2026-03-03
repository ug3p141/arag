# HyProTwin A-RAG Adaptation — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Adapt A-RAG to the HyProTwin WSV inspection report dataset, beating the vanilla RAG baseline on 106 evaluation questions.

**Architecture:** 4-tool agent (metadata_filter, semantic_search, keyword_search, read_document) with Vertex AI/litellm backend, Qdrant vector store, and whole-document reading strategy. See `docs/plans/2026-03-03-hyprotwin-arag-design.md` for full design.

**Tech Stack:** Python 3.10+, litellm, qdrant-client, tiktoken, sentence-transformers, pandas

---

## Task 1: Add Dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add litellm and qdrant-client to core dependencies**

In `pyproject.toml`, replace the `dependencies` list (lines 21-27) with:

```python
dependencies = [
    "requests>=2.28.0",
    "tiktoken>=0.5.0",
    "pyyaml>=6.0",
    "tqdm>=4.65.0",
    "numpy>=1.24.0",
    "litellm>=1.30.0",
    "qdrant-client>=1.7.0",
]
```

Also add `pandas` to the `full` optional dependencies (lines 30-34) if not already present.

**Step 2: Install dependencies**

Run: `uv sync`

**Step 3: Verify imports work**

Run: `uv run python -c "import litellm; import qdrant_client; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add litellm and qdrant-client dependencies"
```

---

## Task 2: LLM Client — Vertex AI via litellm

**Files:**
- Modify: `src/arag/core/llm.py`
- Test: `tests/test_llm_litellm.py`

**Step 1: Write the failing test**

Create `tests/test_llm_litellm.py`:

```python
"""Tests for litellm-based LLM client."""

from unittest.mock import patch, MagicMock
from arag.core.llm import LLMClient


def test_llm_client_init_with_vertex():
    """LLMClient accepts Vertex AI config without requiring API key."""
    client = LLMClient(
        model="vertex_ai/claude-sonnet-4-5-20250929",
        vertex_project="hyprotwin",
        vertex_location="europe-west4",
    )
    assert client.model == "vertex_ai/claude-sonnet-4-5-20250929"


def test_llm_client_init_backward_compat():
    """LLMClient still works with api_key for non-Vertex models."""
    client = LLMClient(model="gpt-4o-mini", api_key="sk-test")
    assert client.model == "gpt-4o-mini"


@patch("arag.core.llm.litellm")
def test_chat_calls_litellm(mock_litellm):
    """chat() delegates to litellm.completion."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.model_dump.return_value = {
        "role": "assistant",
        "content": "Hello",
        "tool_calls": None,
    }
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15
    mock_response.model = "claude-sonnet-4-5-20250929"
    mock_litellm.completion.return_value = mock_response

    client = LLMClient(
        model="vertex_ai/claude-sonnet-4-5-20250929",
        vertex_project="hyprotwin",
        vertex_location="europe-west4",
    )
    result = client.chat(messages=[{"role": "user", "content": "Hi"}])

    mock_litellm.completion.assert_called_once()
    call_kwargs = mock_litellm.completion.call_args
    assert call_kwargs.kwargs["model"] == "vertex_ai/claude-sonnet-4-5-20250929"
    assert result["message"]["content"] == "Hello"


@patch("arag.core.llm.litellm")
def test_chat_with_tools(mock_litellm):
    """chat() passes tools to litellm."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.model_dump.return_value = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "tc_1", "function": {"name": "test", "arguments": "{}"}}],
    }
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15
    mock_response.model = "claude-sonnet-4-5-20250929"
    mock_litellm.completion.return_value = mock_response

    client = LLMClient(
        model="vertex_ai/claude-sonnet-4-5-20250929",
        vertex_project="hyprotwin",
        vertex_location="europe-west4",
    )
    tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
    result = client.chat(
        messages=[{"role": "user", "content": "Hi"}],
        tools=tools,
    )

    call_kwargs = mock_litellm.completion.call_args.kwargs
    assert call_kwargs["tools"] == tools
    assert result["message"]["tool_calls"] is not None


def test_count_tokens():
    """Token counting works without API credentials."""
    client = LLMClient.__new__(LLMClient)
    import tiktoken
    client.tokenizer = tiktoken.encoding_for_model("gpt-4o")
    assert client.count_tokens("Hello world") > 0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_llm_litellm.py -v`
Expected: FAIL — current `LLMClient.__init__` requires `api_key`

**Step 3: Rewrite `src/arag/core/llm.py`**

Replace the entire file with litellm-based implementation:

```python
"""LLM client for ARAG — unified interface via litellm."""

import os
from typing import Any, Dict, List, Optional

import litellm
import tiktoken


class LLMClient:
    """Unified LLM client using litellm for multi-provider support."""

    # Pricing (USD per 1M tokens): (input, cached_input, output)
    PRICING = {
        "claude-sonnet-4-5": (3.0, 0.3, 15.0),
        "claude-opus-4-6": (5.0, 0.5, 25.0),
        "claude-haiku-4-5": (1.0, 0.1, 5.0),
        "gemini-2.5-flash": (0.3, 0.075, 2.5),
        "gemini-2.5-pro": (1.25, 0.125, 10.0),
        "gpt-4o-mini": (0.15, 0.075, 0.6),
        "gpt-4o": (2.5, 1.25, 10.0),
        "default": (1.0, 0.1, 5.0),
    }

    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        base_url: str = None,
        temperature: float = 0.0,
        max_tokens: int = 16384,
        reasoning_effort: str = None,
        vertex_project: str = None,
        vertex_location: str = None,
        vertex_credentials: str = None,
    ):
        self.model = model or os.getenv("ARAG_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.total_cost = 0.0

        # Vertex AI config
        self._litellm_kwargs = {}
        if vertex_project or os.getenv("VERTEX_PROJECT"):
            self._litellm_kwargs["vertex_project"] = vertex_project or os.getenv("VERTEX_PROJECT")
        if vertex_location or os.getenv("VERTEX_LOCATION"):
            self._litellm_kwargs["vertex_location"] = vertex_location or os.getenv("VERTEX_LOCATION")
        if vertex_credentials or os.getenv("VERTEX_CREDENTIALS"):
            self._litellm_kwargs["vertex_credentials"] = vertex_credentials or os.getenv("VERTEX_CREDENTIALS")

        # Non-Vertex: pass API key and base URL
        if api_key or os.getenv("ARAG_API_KEY"):
            self._litellm_kwargs["api_key"] = api_key or os.getenv("ARAG_API_KEY")
        if base_url or os.getenv("ARAG_BASE_URL"):
            self._litellm_kwargs["api_base"] = (base_url or os.getenv("ARAG_BASE_URL")).rstrip("/")

        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def count_message_tokens(self, messages: List[Dict[str, Any]]) -> int:
        total = 0
        for msg in messages:
            total += 4
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.count_tokens(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        total += self.count_tokens(item.get("text", ""))
            if "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    total += self.count_tokens(str(tc.get("function", {})))
        return total

    def calculate_cost(self, usage: dict) -> float:
        """Calculate API cost from usage info."""
        model_lower = self.model.lower()
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        input_tokens = max(prompt_tokens - cached_tokens, 0)

        for key in self.PRICING:
            if key in model_lower:
                input_price, cached_price, output_price = self.PRICING[key]
                break
        else:
            input_price, cached_price, output_price = self.PRICING["default"]

        usd_cost = (
            (input_tokens / 1_000_000) * input_price
            + (cached_tokens / 1_000_000) * cached_price
            + (completion_tokens / 1_000_000) * output_price
        )
        return round(usd_cost, 6)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] = None,
        temperature: float = None,
        max_tokens: int = None,
    ) -> Dict[str, Any]:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            **self._litellm_kwargs,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort

        response = litellm.completion(**kwargs)

        # Parse response into A-RAG's expected format
        message = response.choices[0].message.model_dump()
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }
        cost = self.calculate_cost(usage)
        self.total_cost += cost

        return {
            "message": message,
            "input_tokens": usage["prompt_tokens"],
            "output_tokens": usage["completion_tokens"],
            "cost": cost,
        }

    def generate(
        self,
        messages: List[Dict[str, Any]],
        system: str = None,
        tools: List[Dict[str, Any]] = None,
        temperature: float = None,
        **kwargs,
    ) -> tuple:
        """Generate response (compatibility method for eval script)."""
        if system:
            messages = [{"role": "system", "content": system}] + messages
        result = self.chat(messages=messages, tools=tools, temperature=temperature)
        content = result["message"].get("content", "")
        cost = result["cost"]
        return content, cost
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_llm_litellm.py -v`
Expected: All PASS

**Step 5: Run existing tests to verify backward compatibility**

Run: `uv run pytest tests/test_import.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/arag/core/llm.py tests/test_llm_litellm.py
git commit -m "feat: replace OpenAI HTTP client with litellm for multi-provider support"
```

---

## Task 3: Metadata Extraction from Filenames

**Files:**
- Create: `src/arag/preprocessing/__init__.py`
- Create: `src/arag/preprocessing/metadata.py`
- Test: `tests/test_metadata.py`

**Step 1: Write the failing test**

Create `tests/test_metadata.py`:

```python
"""Tests for metadata extraction from WSV filenames."""

from arag.preprocessing.metadata import extract_metadata_from_filename, STRUCTURE_LOOKUP


def test_bericht_standard_pattern():
    """Parse standard inspection report filename."""
    meta = extract_metadata_from_filename(
        "Bericht_6120_SBL_TR_18.02.2010_complete.md",
        directory_name="30_Wehranlage_Marktbreit_213-6326002",
    )
    assert meta["structure_id"] == "6120"
    assert meta["report_type"] == "SBL_TR"
    assert meta["report_date"] == "2010-02-18"
    assert meta["structure_name"] == "Wehranlage Marktbreit"
    assert meta["doc_type"] == "bericht"


def test_bericht_sbl2_pattern():
    """Parse SBL2 report filename."""
    meta = extract_metadata_from_filename(
        "Bericht_6120_SBL2_SR_04.01.2022_complete.md",
        directory_name="30_Wehranlage_Marktbreit_213-6326002",
    )
    assert meta["structure_id"] == "6120"
    assert meta["report_type"] == "SBL2_SR"
    assert meta["report_date"] == "2022-01-04"


def test_bericht_with_extra_segment():
    """Parse filename with extra segment like '2-33'."""
    meta = extract_metadata_from_filename(
        "Bericht_6120_2-33_TR_05.01.2009_complete.md",
        directory_name="30_Wehranlage_Marktbreit_213-6326002",
    )
    assert meta["structure_id"] == "6120"
    assert meta["report_date"] == "2009-01-05"


def test_besichtigungsbericht_pattern():
    """Parse visit report filename."""
    meta = extract_metadata_from_filename(
        "2014_12_12_Besichtigungsbericht_2136326002_complete.md",
        directory_name="30_Wehranlage_Marktbreit_213-6326002",
    )
    assert meta["report_date"] == "2014-12-12"
    assert meta["object_id"] == "2136326002"
    assert meta["doc_type"] == "besichtigungsbericht"
    assert meta["structure_name"] == "Wehranlage Marktbreit"


def test_numeric_code_pattern():
    """Parse numeric code filename like 612-0000884-0000."""
    meta = extract_metadata_from_filename(
        "612-0000884-0000_complete.md",
        directory_name="30_Wehranlage_Marktbreit_213-6326002",
    )
    assert meta["doc_type"] == "archive"
    assert meta["archive_code"] == "612-0000884-0000"
    assert meta["structure_name"] == "Wehranlage Marktbreit"


def test_baw_report_pattern():
    """Parse BAW annual report filename."""
    meta = extract_metadata_from_filename(
        "95100_BAW_JHWSVPR_Gesamt_Bericht_complete.md",
        directory_name="30_Wehranlage_Marktbreit_213-6326002",
    )
    assert meta["doc_type"] == "baw_report"
    assert meta["structure_name"] == "Wehranlage Marktbreit"


def test_altenrheine_structure():
    """Parse filename from Altenrheine directory."""
    meta = extract_metadata_from_filename(
        "Bericht_4140_SBL2_BU_20.01.2010_complete.md",
        directory_name="39_Schiffsschleusenanlage_Altenrheine_311-3710006",
    )
    assert meta["structure_id"] == "4140"
    assert meta["structure_name"] == "Schiffsschleusenanlage Altenrheine"


def test_structure_lookup_table():
    """Lookup table maps directory names to structure names."""
    assert "Wehranlage Marktbreit" in STRUCTURE_LOOKUP.values()
    assert "Schiffsschleusenanlage Altenrheine" in STRUCTURE_LOOKUP.values()


def test_filename_with_space_suffix():
    """Parse filename with space+number suffix (duplicate variant)."""
    meta = extract_metadata_from_filename(
        "Bericht_6120_SBL2_GM_26.01.2010 2_complete.md",
        directory_name="30_Wehranlage_Marktbreit_213-6326002",
    )
    assert meta["structure_id"] == "6120"
    assert meta["report_type"] == "SBL2_GM"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_metadata.py -v`
Expected: FAIL — module not found

**Step 3: Implement metadata extraction**

Create `src/arag/preprocessing/__init__.py`:

```python
"""Preprocessing modules for A-RAG."""
```

Create `src/arag/preprocessing/metadata.py`:

```python
"""Extract structured metadata from WSV inspection report filenames."""

import re
from typing import Dict, Optional


# Manual lookup: directory name fragment -> structure name
STRUCTURE_LOOKUP = {
    "Wehranlage_Marktbreit": "Wehranlage Marktbreit",
    "Schiffsschleusenanlage_Altenrheine": "Schiffsschleusenanlage Altenrheine",
}

# Regex patterns for filename types
_BERICHT_RE = re.compile(
    r"^Bericht_(\d+)_(.+?)_(\d{2}\.\d{2}\.\d{4})"
    r"(?:\s+\d+)?"  # optional space+digit suffix for duplicates
    r"_complete\.md$"
)
_BESICHTIGUNG_RE = re.compile(
    r"^(\d{4})_(\d{2})_(\d{2})_Besichtigungsbericht_(\d+)"
    r"_complete\.md$"
)
_ARCHIVE_RE = re.compile(
    r"^(\d{3}-\d{7}-\d{4})_complete\.md$"
)
_BAW_RE = re.compile(
    r"^95100_BAW_.+_complete\.md$"
)


def _parse_date_dmy(date_str: str) -> str:
    """Convert dd.mm.yyyy to yyyy-mm-dd."""
    parts = date_str.split(".")
    return f"{parts[2]}-{parts[1]}-{parts[0]}"


def _structure_name_from_directory(directory_name: str) -> str:
    """Extract structure name from directory name via lookup table."""
    for key, name in STRUCTURE_LOOKUP.items():
        if key in directory_name:
            return name
    # Fallback: parse directory name (strip leading number and trailing ID)
    parts = directory_name.split("_", 1)
    if len(parts) > 1:
        return parts[1].rsplit("_", 1)[0].replace("_", " ")
    return directory_name


def extract_metadata_from_filename(
    filename: str,
    directory_name: str = "",
) -> Dict[str, Optional[str]]:
    """Extract metadata from a WSV document filename.

    Args:
        filename: The document filename (e.g. 'Bericht_6120_SBL_TR_18.02.2010_complete.md')
        directory_name: Parent directory name (e.g. '30_Wehranlage_Marktbreit_213-6326002')

    Returns:
        Dict with extracted metadata fields.
    """
    meta: Dict[str, Optional[str]] = {
        "filename": filename,
        "structure_name": _structure_name_from_directory(directory_name),
    }

    # Try Bericht pattern
    m = _BERICHT_RE.match(filename)
    if m:
        meta["doc_type"] = "bericht"
        meta["structure_id"] = m.group(1)
        meta["report_type"] = m.group(2)
        meta["report_date"] = _parse_date_dmy(m.group(3))
        return meta

    # Try Besichtigungsbericht pattern
    m = _BESICHTIGUNG_RE.match(filename)
    if m:
        meta["doc_type"] = "besichtigungsbericht"
        meta["report_date"] = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        meta["object_id"] = m.group(4)
        return meta

    # Try archive code pattern
    m = _ARCHIVE_RE.match(filename)
    if m:
        meta["doc_type"] = "archive"
        meta["archive_code"] = m.group(1)
        return meta

    # Try BAW report pattern
    m = _BAW_RE.match(filename)
    if m:
        meta["doc_type"] = "baw_report"
        return meta

    # Unknown pattern
    meta["doc_type"] = "unknown"
    return meta
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_metadata.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/arag/preprocessing/ tests/test_metadata.py
git commit -m "feat: add metadata extraction from WSV filenames"
```

---

## Task 4: Structure-Aware Chunking Pipeline

**Files:**
- Create: `src/arag/preprocessing/chunker.py`
- Test: `tests/test_chunker.py`

**Step 1: Write the failing test**

Create `tests/test_chunker.py`:

```python
"""Tests for structure-aware chunking of WSV inspection reports."""

from arag.preprocessing.chunker import chunk_document


SAMPLE_DOC = """## Page 1

Wasser- und Schifffahrtsverwaltung des Bundes

Objektidentnr.: 2136326002
WaStr.-km: 275.700
Prüfbericht Wehranlage Marktbreit linkes Wehrfeld
Inspektionsjahr: 2009
Baujahr: 1954
Prüfnote: 2.9

## Page 2

1 linkes Wehrfeld, Walze Zahnleiste
Metall, Verschleiß
Beschreibung des Schadens an der Zahnleiste. SK 2

![img-0.jpeg](img-0.jpeg)

img-0-description: Foto zeigt die Zahnleiste mit Verschleiß.

2 linkes Wehrfeld, Walze Beschichtung
Metall, Korrosion
Flächige Korrosion mit beginnendem Blattrost. SK 3

![img-1.jpeg](img-1.jpeg)

img-1-description: Korrosion an der Walzenbeschichtung.

## Page 5

Beurteilung und Veranlassung
Gesamtnote 2.9. Instandsetzung empfohlen.
"""


def test_chunk_document_returns_list():
    """chunk_document returns a list of chunk dicts."""
    chunks = chunk_document(
        text=SAMPLE_DOC,
        document_id="6120_2009_SBL_TR",
        metadata={"structure_name": "Wehranlage Marktbreit", "inspection_year": 2009},
    )
    assert isinstance(chunks, list)
    assert len(chunks) > 0


def test_chunk_has_required_fields():
    """Each chunk has id, text, and metadata fields."""
    chunks = chunk_document(
        text=SAMPLE_DOC,
        document_id="6120_2009_SBL_TR",
        metadata={"structure_name": "Wehranlage Marktbreit"},
    )
    for chunk in chunks:
        assert "id" in chunk
        assert "text" in chunk
        assert chunk["id"].startswith("6120_2009_SBL_TR_")


def test_contextual_header_prepended():
    """Each chunk text starts with a contextual header."""
    chunks = chunk_document(
        text=SAMPLE_DOC,
        document_id="6120_2009_SBL_TR",
        metadata={"structure_name": "Wehranlage Marktbreit", "inspection_year": 2009},
    )
    for chunk in chunks:
        assert chunk["text"].startswith("[")


def test_kopfdaten_chunk_is_first():
    """First chunk contains header/Kopfdaten information."""
    chunks = chunk_document(
        text=SAMPLE_DOC,
        document_id="6120_2009_SBL_TR",
        metadata={"structure_name": "Wehranlage Marktbreit"},
    )
    assert "Objektidentnr" in chunks[0]["text"] or "Prüfbericht" in chunks[0]["text"]


def test_inspection_year_extracted():
    """Inspection year is extracted from document content."""
    from arag.preprocessing.chunker import extract_inspection_year
    year = extract_inspection_year(SAMPLE_DOC, report_date="2010-02-18")
    assert year == 2009
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_chunker.py -v`
Expected: FAIL — module not found

**Step 3: Implement chunker**

Create `src/arag/preprocessing/chunker.py`:

```python
"""Structure-aware chunking for WSV inspection reports."""

import re
from typing import Dict, List, Optional


def extract_inspection_year(text: str, report_date: str = None) -> Optional[int]:
    """Extract inspection year from document text, falling back to report date."""
    # Look for explicit inspection year
    for pattern in [
        r"Inspektionsjahr[:\s]*(\d{4})",
        r"Prüfjahr[:\s]*(\d{4})",
        r"Inspektionsjahr\s*(\d{4})",
    ]:
        m = re.search(pattern, text)
        if m:
            return int(m.group(1))

    # Fallback to report_date year
    if report_date:
        return int(report_date[:4])

    return None


def _build_header(document_id: str, metadata: Dict) -> str:
    """Build contextual header string for a chunk."""
    parts = []
    if metadata.get("structure_name"):
        parts.append(metadata["structure_name"])
    if metadata.get("structure_id"):
        parts.append(f"ID:{metadata['structure_id']}")
    if metadata.get("inspection_year"):
        parts.append(f"Inspektion {metadata['inspection_year']}")
    if metadata.get("report_type"):
        parts.append(metadata["report_type"])
    return f"[{' | '.join(parts)}]" if parts else f"[{document_id}]"


def _split_into_sections(text: str) -> List[Dict[str, str]]:
    """Split markdown text into sections by ## headings and numbered damage entries."""
    sections = []
    current_title = "header"
    current_lines = []

    for line in text.split("\n"):
        # New page/section heading
        if re.match(r"^##\s+", line):
            if current_lines:
                sections.append({"title": current_title, "text": "\n".join(current_lines)})
            current_title = line.strip("# ").strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append({"title": current_title, "text": "\n".join(current_lines)})

    return sections


def chunk_document(
    text: str,
    document_id: str,
    metadata: Dict,
    max_chunk_tokens: int = 1500,
) -> List[Dict]:
    """Chunk a WSV inspection report into semantically meaningful pieces.

    Args:
        text: Full document text (markdown).
        document_id: Document identifier (e.g. '6120_2009_SBL_TR').
        metadata: Dict with structure_name, structure_id, inspection_year, etc.
        max_chunk_tokens: Approximate max tokens per chunk (estimated at 4 chars/token).

    Returns:
        List of chunk dicts with 'id', 'text', 'document_id', and metadata fields.
    """
    header_prefix = _build_header(document_id, metadata)
    sections = _split_into_sections(text)
    chunks = []
    seq = 0
    max_chars = max_chunk_tokens * 4  # rough token estimate

    for section in sections:
        section_text = section["text"].strip()
        if not section_text:
            continue

        chunk_text = f"{header_prefix}\n{section_text}"

        if len(chunk_text) <= max_chars:
            chunks.append({
                "id": f"{document_id}_{seq:02d}",
                "text": chunk_text,
                "document_id": document_id,
                "section": section["title"],
                **{k: v for k, v in metadata.items()},
            })
            seq += 1
        else:
            # Split large sections at paragraph boundaries
            paragraphs = re.split(r"\n\n+", section_text)
            current_batch = []
            current_len = len(header_prefix) + 1

            for para in paragraphs:
                if current_len + len(para) > max_chars and current_batch:
                    chunks.append({
                        "id": f"{document_id}_{seq:02d}",
                        "text": f"{header_prefix}\n" + "\n\n".join(current_batch),
                        "document_id": document_id,
                        "section": section["title"],
                        **{k: v for k, v in metadata.items()},
                    })
                    seq += 1
                    current_batch = []
                    current_len = len(header_prefix) + 1

                current_batch.append(para)
                current_len += len(para) + 2

            if current_batch:
                chunks.append({
                    "id": f"{document_id}_{seq:02d}",
                    "text": f"{header_prefix}\n" + "\n\n".join(current_batch),
                    "document_id": document_id,
                    "section": section["title"],
                    **{k: v for k, v in metadata.items()},
                })
                seq += 1

    return chunks
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_chunker.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/arag/preprocessing/chunker.py tests/test_chunker.py
git commit -m "feat: add structure-aware chunking for WSV inspection reports"
```

---

## Task 5: Preprocessing Script — Build chunks.json and document_registry.json

**Files:**
- Create: `scripts/preprocess_wsv.py`
- Test: manual verification (script produces expected output files)

**Step 1: Implement the preprocessing script**

Create `scripts/preprocess_wsv.py`:

```python
#!/usr/bin/env python3
"""Preprocess WSV inspection reports into A-RAG format.

Scans data/wsv/ for _complete.md files, extracts metadata, chunks documents,
and produces chunks.json + document_registry.json.

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
            "source_file": str(md_file.relative_to(data_dir.parent.parent)),
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
```

**Step 2: Run the script**

Run: `uv run python scripts/preprocess_wsv.py --data-dir data/wsv --output-dir data/wsv`
Expected: Creates `data/wsv/chunks.json` and `data/wsv/document_registry.json`, prints counts.

**Step 3: Verify output**

Run: `uv run python -c "import json; d=json.load(open('data/wsv/chunks.json')); print(f'{len(d)} chunks'); r=json.load(open('data/wsv/document_registry.json')); print(f'{len(r)} documents')"`
Expected: Shows chunk count and ~99 documents.

**Step 4: Commit**

```bash
git add scripts/preprocess_wsv.py data/wsv/chunks.json data/wsv/document_registry.json
git commit -m "feat: add WSV preprocessing script, generate chunks and document registry"
```

---

## Task 6: Eval Catalog Conversion

**Files:**
- Create: `scripts/convert_eval_catalog.py`
- Test: manual verification

**Step 1: Implement the conversion script**

Create `scripts/convert_eval_catalog.py`:

```python
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
                    "vanilla_retrieval_score": float(row["retrieval_score"]) if row["retrieval_score"] else None,
                    "vanilla_precision_at_k": float(row["precision_at_k"]) if row["precision_at_k"] else None,
                },
            })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(questions)} questions to {args.output}")

    # Print category breakdown
    from collections import Counter
    cats = Counter(q["metadata"]["kategorie"] for q in questions)
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
```

**Step 2: Run the script**

Run: `uv run python scripts/convert_eval_catalog.py`
Expected: Creates `data/wsv/questions.json` with 106 questions, prints category breakdown.

**Step 3: Commit**

```bash
git add scripts/convert_eval_catalog.py data/wsv/questions.json
git commit -m "feat: add eval catalog conversion script"
```

---

## Task 7: Tool — `read_document`

**Files:**
- Create: `src/arag/tools/read_document.py`
- Test: `tests/test_read_document.py`

**Step 1: Write the failing test**

Create `tests/test_read_document.py`:

```python
"""Tests for read_document tool."""

import json
import tempfile
from pathlib import Path

from arag.core.context import AgentContext
from arag.tools.read_document import ReadDocumentTool


def _make_registry(tmp_path: Path) -> tuple[Path, Path]:
    """Create a minimal document registry and document file for testing."""
    doc_file = tmp_path / "test_doc_complete.md"
    doc_file.write_text("# Test Document\n\nSome content about damages.\n", encoding="utf-8")

    registry = {
        "6120_2009_SBL_TR": {
            "source_file": str(doc_file),
            "structure_name": "Wehranlage Marktbreit",
            "inspection_year": 2009,
        }
    }
    registry_file = tmp_path / "document_registry.json"
    registry_file.write_text(json.dumps(registry), encoding="utf-8")

    return registry_file, doc_file


def test_read_document_returns_content(tmp_path):
    """read_document returns full document text."""
    registry_file, _ = _make_registry(tmp_path)
    tool = ReadDocumentTool(registry_file=str(registry_file))
    ctx = AgentContext()

    result, log = tool.execute(ctx, document_id="6120_2009_SBL_TR")

    assert "Test Document" in result
    assert "Some content about damages" in result
    assert log["retrieved_tokens"] > 0


def test_read_document_unknown_id(tmp_path):
    """read_document returns error for unknown document_id."""
    registry_file, _ = _make_registry(tmp_path)
    tool = ReadDocumentTool(registry_file=str(registry_file))
    ctx = AgentContext()

    result, log = tool.execute(ctx, document_id="nonexistent")

    assert "not found" in result.lower()


def test_read_document_marks_read(tmp_path):
    """read_document marks the document as read in context."""
    registry_file, _ = _make_registry(tmp_path)
    tool = ReadDocumentTool(registry_file=str(registry_file))
    ctx = AgentContext()

    tool.execute(ctx, document_id="6120_2009_SBL_TR")

    assert ctx.is_chunk_read("doc:6120_2009_SBL_TR")


def test_read_document_skips_reread(tmp_path):
    """Second read of same document returns short message."""
    registry_file, _ = _make_registry(tmp_path)
    tool = ReadDocumentTool(registry_file=str(registry_file))
    ctx = AgentContext()

    tool.execute(ctx, document_id="6120_2009_SBL_TR")
    result2, log2 = tool.execute(ctx, document_id="6120_2009_SBL_TR")

    assert "already been read" in result2.lower()
    assert log2["retrieved_tokens"] == 0


def test_tool_name():
    """Tool name is 'read_document'."""
    tool = ReadDocumentTool.__new__(ReadDocumentTool)
    assert tool.name == "read_document"


def test_tool_schema():
    """Schema has required fields."""
    tool = ReadDocumentTool.__new__(ReadDocumentTool)
    schema = tool.get_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "read_document"
    assert "document_id" in schema["function"]["parameters"]["properties"]
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_read_document.py -v`
Expected: FAIL — module not found

**Step 3: Implement read_document tool**

Create `src/arag/tools/read_document.py`:

```python
"""Read full document content by document ID."""

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import tiktoken

from arag.tools.base import BaseTool
from arag.core.context import AgentContext


class ReadDocumentTool(BaseTool):
    """Read the full text of an inspection report by document ID."""

    def __init__(self, registry_file: str):
        with open(registry_file, "r", encoding="utf-8") as f:
            self._registry: Dict[str, Dict] = json.load(f)
        self._tokenizer = tiktoken.encoding_for_model("gpt-4o")

    @property
    def name(self) -> str:
        return "read_document"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "read_document",
                "description": (
                    "Read the full text of an inspection report by its document ID. "
                    "Use after identifying relevant documents via metadata_filter, "
                    "semantic_search, or keyword_search. Returns the complete document "
                    "content for in-depth analysis."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": (
                                "The document identifier, e.g. '6120_2009_SBL_TR'. "
                                "Obtain this from the document_id field in search results."
                            ),
                        },
                    },
                    "required": ["document_id"],
                },
            },
        }

    def execute(self, context: AgentContext, **kwargs) -> Tuple[str, Dict[str, Any]]:
        document_id = kwargs.get("document_id", "")
        read_key = f"doc:{document_id}"

        # Check if already read
        if context.is_chunk_read(read_key):
            return (
                f"Document '{document_id}' has already been read in this session.",
                {"retrieved_tokens": 0, "already_read": True},
            )

        # Look up in registry
        entry = self._registry.get(document_id)
        if not entry:
            return (
                f"Document ID '{document_id}' not found in the registry. "
                f"Available documents can be found via metadata_filter or search tools.",
                {"retrieved_tokens": 0, "error": "not_found"},
            )

        # Read the file
        source_file = Path(entry["source_file"])
        if not source_file.exists():
            # Try relative to working directory
            source_file = Path(entry["source_file"])
            if not source_file.exists():
                return (
                    f"Source file not found: {entry['source_file']}",
                    {"retrieved_tokens": 0, "error": "file_not_found"},
                )

        text = source_file.read_text(encoding="utf-8")
        tokens = len(self._tokenizer.encode(text))

        # Mark as read
        context.mark_chunk_as_read(read_key)
        context.add_retrieval_log(
            tool_name="read_document",
            retrieved_tokens=tokens,
            metadata={"document_id": document_id},
        )

        # Build response with metadata header
        header_parts = [f"Document: {document_id}"]
        if entry.get("structure_name"):
            header_parts.append(f"Structure: {entry['structure_name']}")
        if entry.get("inspection_year"):
            header_parts.append(f"Inspection Year: {entry['inspection_year']}")
        header = " | ".join(header_parts)

        return (
            f"{'='*80}\n{header}\n{'='*80}\n\n{text}",
            {"retrieved_tokens": tokens, "document_id": document_id},
        )
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_read_document.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/arag/tools/read_document.py tests/test_read_document.py
git commit -m "feat: add read_document tool for full document retrieval"
```

---

## Task 8: Tool — `metadata_filter`

**Files:**
- Create: `src/arag/tools/metadata_filter.py`
- Test: `tests/test_metadata_filter.py`

**Step 1: Write the failing test**

Create `tests/test_metadata_filter.py`:

```python
"""Tests for metadata_filter tool (uses document registry, no Qdrant for unit tests)."""

import json
from pathlib import Path

from arag.core.context import AgentContext
from arag.tools.metadata_filter import MetadataFilterTool


def _make_registry(tmp_path: Path) -> Path:
    registry = {
        "6120_2009_SBL_TR": {
            "source_file": "data/wsv/30_.../file.md",
            "structure_name": "Wehranlage Marktbreit",
            "structure_id": "6120",
            "inspection_year": 2009,
            "report_type": "SBL_TR",
            "doc_type": "bericht",
        },
        "6120_2018_SBL2_SR": {
            "source_file": "data/wsv/30_.../file2.md",
            "structure_name": "Wehranlage Marktbreit",
            "structure_id": "6120",
            "inspection_year": 2018,
            "report_type": "SBL2_SR",
            "doc_type": "bericht",
        },
        "4140_2010_SBL2_BU": {
            "source_file": "data/wsv/39_.../file3.md",
            "structure_name": "Schiffsschleusenanlage Altenrheine",
            "structure_id": "4140",
            "inspection_year": 2010,
            "report_type": "SBL2_BU",
            "doc_type": "bericht",
        },
    }
    path = tmp_path / "document_registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    return path


def test_filter_by_structure_name(tmp_path):
    """Filter by structure_name returns matching documents."""
    tool = MetadataFilterTool(registry_file=str(_make_registry(tmp_path)))
    ctx = AgentContext()
    result, log = tool.execute(ctx, structure_name="Marktbreit")
    assert "6120_2009_SBL_TR" in result
    assert "6120_2018_SBL2_SR" in result
    assert "4140" not in result


def test_filter_by_year(tmp_path):
    """Filter by inspection_year returns matching documents."""
    tool = MetadataFilterTool(registry_file=str(_make_registry(tmp_path)))
    ctx = AgentContext()
    result, log = tool.execute(ctx, inspection_year=2009)
    assert "6120_2009_SBL_TR" in result
    assert "2018" not in result


def test_filter_combined(tmp_path):
    """Combining filters narrows results."""
    tool = MetadataFilterTool(registry_file=str(_make_registry(tmp_path)))
    ctx = AgentContext()
    result, log = tool.execute(ctx, structure_name="Marktbreit", inspection_year=2018)
    assert "6120_2018_SBL2_SR" in result
    assert "6120_2009" not in result


def test_filter_no_match(tmp_path):
    """No matches returns informative message."""
    tool = MetadataFilterTool(registry_file=str(_make_registry(tmp_path)))
    ctx = AgentContext()
    result, log = tool.execute(ctx, structure_name="Nonexistent")
    assert "no documents" in result.lower() or log["documents_found"] == 0


def test_filter_no_params(tmp_path):
    """No filter parameters returns error."""
    tool = MetadataFilterTool(registry_file=str(_make_registry(tmp_path)))
    ctx = AgentContext()
    result, log = tool.execute(ctx)
    assert "at least one" in result.lower()


def test_tool_schema(tmp_path):
    """Schema includes all filter parameters."""
    tool = MetadataFilterTool(registry_file=str(_make_registry(tmp_path)))
    schema = tool.get_schema()
    params = schema["function"]["parameters"]["properties"]
    assert "structure_name" in params
    assert "structure_id" in params
    assert "inspection_year" in params
    assert "report_type" in params
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_metadata_filter.py -v`
Expected: FAIL — module not found

**Step 3: Implement metadata_filter tool**

Create `src/arag/tools/metadata_filter.py`:

```python
"""Filter documents by structured metadata from the document registry."""

import json
from typing import Any, Dict, Tuple

from arag.tools.base import BaseTool
from arag.core.context import AgentContext


class MetadataFilterTool(BaseTool):
    """Filter inspection reports by structured metadata fields."""

    def __init__(self, registry_file: str):
        with open(registry_file, "r", encoding="utf-8") as f:
            self._registry: Dict[str, Dict] = json.load(f)

    @property
    def name(self) -> str:
        return "metadata_filter"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "metadata_filter",
                "description": (
                    "Filter inspection report documents by structured metadata. "
                    "Use when the question mentions a specific structure name, "
                    "structure ID, inspection year, or report type. "
                    "Returns a list of matching documents with their metadata. "
                    "Then use read_document to read the full content of a matching document."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "structure_name": {
                            "type": "string",
                            "description": (
                                "Structure name or partial name to match, e.g. 'Marktbreit' "
                                "or 'Wehranlage Marktbreit'. Case-insensitive substring match."
                            ),
                        },
                        "structure_id": {
                            "type": "string",
                            "description": "Exact structure ID, e.g. '6120', '4140'.",
                        },
                        "inspection_year": {
                            "type": "integer",
                            "description": "Inspection year, e.g. 2009, 2018.",
                        },
                        "report_type": {
                            "type": "string",
                            "description": "Report type code, e.g. 'SBL_TR', 'SBL2_SR', 'SBL2_GM'.",
                        },
                    },
                },
            },
        }

    def execute(self, context: AgentContext, **kwargs) -> Tuple[str, Dict[str, Any]]:
        structure_name = kwargs.get("structure_name")
        structure_id = kwargs.get("structure_id")
        inspection_year = kwargs.get("inspection_year")
        report_type = kwargs.get("report_type")

        if not any([structure_name, structure_id, inspection_year, report_type]):
            return (
                "Error: At least one filter parameter required "
                "(structure_name, structure_id, inspection_year, or report_type).",
                {"documents_found": 0, "error": "no_parameters"},
            )

        matches = []
        for doc_id, entry in self._registry.items():
            if structure_name:
                entry_name = (entry.get("structure_name") or "").lower()
                if structure_name.lower() not in entry_name:
                    continue
            if structure_id:
                if entry.get("structure_id") != structure_id:
                    continue
            if inspection_year is not None:
                if entry.get("inspection_year") != inspection_year:
                    continue
            if report_type:
                if entry.get("report_type") != report_type:
                    continue
            matches.append((doc_id, entry))

        if not matches:
            return (
                "No documents match the given filters.",
                {"documents_found": 0},
            )

        # Format results
        lines = [f"Found {len(matches)} matching document(s):\n"]
        for doc_id, entry in matches:
            parts = [f"  document_id: {doc_id}"]
            if entry.get("structure_name"):
                parts.append(f"  structure: {entry['structure_name']}")
            if entry.get("inspection_year"):
                parts.append(f"  inspection_year: {entry['inspection_year']}")
            if entry.get("report_type"):
                parts.append(f"  report_type: {entry['report_type']}")
            if entry.get("doc_type"):
                parts.append(f"  doc_type: {entry['doc_type']}")
            lines.append("\n".join(parts))
            lines.append("")

        context.add_retrieval_log(
            tool_name="metadata_filter",
            retrieved_tokens=0,
            metadata={"documents_found": len(matches), "filters": kwargs},
        )

        return (
            "\n".join(lines),
            {"documents_found": len(matches)},
        )
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_metadata_filter.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/arag/tools/metadata_filter.py tests/test_metadata_filter.py
git commit -m "feat: add metadata_filter tool for structured document filtering"
```

---

## Task 9: Adapt semantic_search for Qdrant

**Files:**
- Create: `src/arag/tools/semantic_search_qdrant.py`
- Test: `tests/test_semantic_search_qdrant.py`

**Step 1: Write the failing test**

Create `tests/test_semantic_search_qdrant.py`:

```python
"""Tests for Qdrant-based semantic search tool."""

from unittest.mock import MagicMock, patch
from arag.core.context import AgentContext
from arag.tools.semantic_search_qdrant import QdrantSemanticSearchTool


def _make_mock_qdrant():
    """Create a mock Qdrant client with search results."""
    client = MagicMock()
    point = MagicMock()
    point.id = "6120_2009_SBL_TR_01"
    point.score = 0.85
    point.payload = {
        "text": "Korrosion an der Walzenbeschichtung mit beginnendem Blattrost.",
        "document_id": "6120_2009_SBL_TR",
        "structure_name": "Wehranlage Marktbreit",
        "inspection_year": 2009,
    }
    client.search.return_value = [point]
    return client


def test_semantic_search_returns_results():
    """Search returns formatted results with document_id."""
    client = _make_mock_qdrant()
    embed_fn = MagicMock(return_value=[0.1] * 384)
    tool = QdrantSemanticSearchTool(
        qdrant_client=client,
        collection_name="test",
        embedding_fn=embed_fn,
    )
    ctx = AgentContext()
    result, log = tool.execute(ctx, query="Korrosion Walze", top_k=5)

    assert "6120_2009_SBL_TR" in result
    assert "Korrosion" in result
    assert log["chunks_found"] == 1
    client.search.assert_called_once()


def test_semantic_search_schema():
    """Schema has query and top_k parameters."""
    tool = QdrantSemanticSearchTool.__new__(QdrantSemanticSearchTool)
    schema = tool.get_schema()
    params = schema["function"]["parameters"]["properties"]
    assert "query" in params
    assert "top_k" in params


def test_tool_name():
    tool = QdrantSemanticSearchTool.__new__(QdrantSemanticSearchTool)
    assert tool.name == "semantic_search"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_semantic_search_qdrant.py -v`
Expected: FAIL — module not found

**Step 3: Implement Qdrant semantic search**

Create `src/arag/tools/semantic_search_qdrant.py`:

```python
"""Semantic search via Qdrant vector database."""

from typing import Any, Callable, Dict, List, Tuple

import tiktoken

from arag.tools.base import BaseTool
from arag.core.context import AgentContext


class QdrantSemanticSearchTool(BaseTool):
    """Semantic search over inspection report chunks using Qdrant."""

    def __init__(
        self,
        qdrant_client,
        collection_name: str,
        embedding_fn: Callable[[str], List[float]],
    ):
        self._client = qdrant_client
        self._collection = collection_name
        self._embed = embedding_fn
        self._tokenizer = tiktoken.encoding_for_model("gpt-4o")

    @property
    def name(self) -> str:
        return "semantic_search"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "semantic_search",
                "description": (
                    "Search inspection report chunks by semantic similarity. "
                    "Use for conceptual queries where exact keywords are unknown, "
                    "e.g. 'Instandsetzungsmaßnahmen', 'Korrosionsschäden'. "
                    "Returns matching chunks with their document_id for follow-up "
                    "with read_document."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query in German.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return (default 10, max 20).",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def execute(self, context: AgentContext, **kwargs) -> Tuple[str, Dict[str, Any]]:
        query = kwargs.get("query", "")
        top_k = min(kwargs.get("top_k", 10), 20)

        query_vector = self._embed(query)
        results = self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )

        if not results:
            return "No matching chunks found.", {"chunks_found": 0, "retrieved_tokens": 0}

        # Deduplicate by document_id, keep best score per document
        seen_docs = {}
        lines = []
        total_tokens = 0

        for r in results:
            doc_id = r.payload.get("document_id", "unknown")
            text_preview = r.payload.get("text", "")[:300]
            tokens = len(self._tokenizer.encode(text_preview))
            total_tokens += tokens

            lines.append(
                f"chunk_id: {r.id}\n"
                f"  document_id: {doc_id}\n"
                f"  score: {r.score:.3f}\n"
                f"  structure: {r.payload.get('structure_name', 'N/A')}\n"
                f"  inspection_year: {r.payload.get('inspection_year', 'N/A')}\n"
                f"  preview: {text_preview}\n"
            )

        context.add_retrieval_log(
            tool_name="semantic_search",
            retrieved_tokens=total_tokens,
            metadata={"query": query, "chunks_found": len(results)},
        )

        return (
            f"Found {len(results)} matching chunks:\n\n" + "\n".join(lines),
            {"chunks_found": len(results), "retrieved_tokens": total_tokens},
        )
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_semantic_search_qdrant.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/arag/tools/semantic_search_qdrant.py tests/test_semantic_search_qdrant.py
git commit -m "feat: add Qdrant-based semantic search tool"
```

---

## Task 10: System Prompt and Config

**Files:**
- Create: `src/arag/agent/prompts/hyprotwin.txt`
- Create: `configs/hyprotwin.yaml`

**Step 1: Write the domain system prompt**

Create `src/arag/agent/prompts/hyprotwin.txt`:

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

AVAILABLE TOOLS:
- metadata_filter: Filter documents by structure name, ID, inspection year, or report type.
- semantic_search: Find chunks by meaning (for conceptual queries).
- keyword_search: Find chunks by exact keyword matching.
- read_document: Read the full text of a document by its document_id.

RETRIEVAL STRATEGY:
1. When the question names a structure and/or year: START with metadata_filter
   to identify the right report(s), then read_document to read them fully.
2. For conceptual queries without specific identifiers: use semantic_search
   to discover relevant chunks, note their document_ids, then read_document.
3. For temporal comparison questions: call metadata_filter for each year,
   read_document for each report, then compare.
4. Use keyword_search for exact terms (Schadenkennung codes, Bauteil names)
   when semantic_search is too broad.
5. ALWAYS read the full document with read_document before answering.
   Do not answer from chunk previews or search result snippets alone.

ANSWER GUIDELINES:
- Cite the specific report (structure name, ID, inspection year).
- When reporting damages, include Schadensklasse, Bauteil, and Schadenkennung.
- If the corpus lacks sufficient information, say so explicitly.
- Respond in the same language as the question (typically German).
```

**Step 2: Write the HyProTwin config**

Create `configs/hyprotwin.yaml`:

```yaml
llm:
  model: "vertex_ai/claude-sonnet-4-5-20250929"
  temperature: 0.0
  max_tokens: 16384
  vertex_project: "hyprotwin"
  vertex_location: "europe-west4"
  vertex_credentials: "vertex_credentials/hyprotwin-58e6370ca989.json"

agent:
  max_loops: 15
  max_token_budget: 200000
  verbose: false
  system_prompt: "src/arag/agent/prompts/hyprotwin.txt"

data:
  chunks_file: "data/wsv/chunks.json"
  document_registry: "data/wsv/document_registry.json"
  questions_file: "data/wsv/questions.json"

qdrant:
  host: "localhost"
  port: 6333
  collection: "hyprotwin"

embedding:
  model: "text-embedding-ada-002"

output:
  results_dir: "results/hyprotwin/"
```

**Step 3: Commit**

```bash
git add src/arag/agent/prompts/hyprotwin.txt configs/hyprotwin.yaml
git commit -m "feat: add HyProTwin system prompt and config"
```

---

## Task 11: Adapt batch_runner for HyProTwin

**Files:**
- Create: `scripts/batch_runner_hyprotwin.py`
- Test: `tests/test_batch_runner_hyprotwin.py`

**Step 1: Write the failing test**

Create `tests/test_batch_runner_hyprotwin.py`:

```python
"""Tests for HyProTwin batch runner initialization."""

from unittest.mock import patch, MagicMock
from arag import Config


def test_config_loads_hyprotwin():
    """HyProTwin config loads without error."""
    config = Config.from_yaml("configs/hyprotwin.yaml")
    assert config.get("llm.model") == "vertex_ai/claude-sonnet-4-5-20250929"
    assert config.get("agent.max_loops") == 15
    assert config.get("data.document_registry") == "data/wsv/document_registry.json"
```

**Step 2: Run test**

Run: `uv run pytest tests/test_batch_runner_hyprotwin.py -v`
Expected: PASS (config file exists from Task 10)

**Step 3: Create the HyProTwin batch runner**

Create `scripts/batch_runner_hyprotwin.py`:

```python
#!/usr/bin/env python3
"""Batch runner adapted for HyProTwin WSV evaluation.

Usage:
    uv run python scripts/batch_runner_hyprotwin.py \
        --config configs/hyprotwin.yaml \
        --output results/hyprotwin/
"""

import argparse
import json
import logging
import os
from pathlib import Path
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from tqdm import tqdm
from arag import LLMClient, BaseAgent, ToolRegistry, Config
from arag.tools.keyword_search import KeywordSearchTool
from arag.tools.read_document import ReadDocumentTool
from arag.tools.metadata_filter import MetadataFilterTool

logging.basicConfig(level=logging.ERROR)


class HyProTwinBatchRunner:
    """Batch runner for HyProTwin WSV evaluation."""

    def __init__(
        self,
        config: Config,
        output_dir: str,
        limit: int = None,
        num_workers: int = 5,
        verbose: bool = False,
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.limit = limit
        self.num_workers = num_workers
        self.verbose = verbose

        self.predictions_file = self.output_dir / "predictions.jsonl"
        self.write_lock = Lock()

        # Load questions
        questions_file = config.get("data.questions_file", "data/wsv/questions.json")
        with open(questions_file, "r", encoding="utf-8") as f:
            self.questions = json.load(f)
        if self.limit:
            self.questions = self.questions[: self.limit]

        # Load system prompt
        prompt_file = config.get("agent.system_prompt", "src/arag/agent/prompts/hyprotwin.txt")
        self._system_prompt = Path(prompt_file).read_text(encoding="utf-8")

        # Initialize shared tools
        self._shared_tools = self._init_tools()

    def _init_tools(self) -> ToolRegistry:
        data_config = self.config.get("data", {})
        chunks_file = data_config.get("chunks_file", "data/wsv/chunks.json")
        registry_file = data_config.get("document_registry", "data/wsv/document_registry.json")

        tools = ToolRegistry()
        tools.register(KeywordSearchTool(chunks_file=chunks_file))
        tools.register(ReadDocumentTool(registry_file=registry_file))
        tools.register(MetadataFilterTool(registry_file=registry_file))

        # Qdrant semantic search (optional — skip if not configured)
        qdrant_config = self.config.get("qdrant", {})
        if qdrant_config.get("host"):
            try:
                from qdrant_client import QdrantClient
                from arag.tools.semantic_search_qdrant import QdrantSemanticSearchTool

                client = QdrantClient(
                    host=qdrant_config["host"],
                    port=qdrant_config.get("port", 6333),
                )
                embedding_config = self.config.get("embedding", {})

                # Build embedding function via litellm
                import litellm

                llm_config = self.config.get("llm", {})

                def embed_fn(text: str) -> list[float]:
                    resp = litellm.embedding(
                        model=embedding_config.get("model", "text-embedding-ada-002"),
                        input=[text],
                        vertex_project=llm_config.get("vertex_project"),
                        vertex_location=llm_config.get("vertex_location"),
                        vertex_credentials=llm_config.get("vertex_credentials"),
                    )
                    return resp.data[0]["embedding"]

                tools.register(
                    QdrantSemanticSearchTool(
                        qdrant_client=client,
                        collection_name=qdrant_config.get("collection", "hyprotwin"),
                        embedding_fn=embed_fn,
                    )
                )
                print("Qdrant semantic search enabled.")
            except Exception as e:
                print(f"Qdrant semantic search disabled: {e}")

        print(f"Registered tools: {tools.list_tools()}")
        return tools

    def _create_agent(self) -> BaseAgent:
        llm_config = self.config.get("llm", {})
        client = LLMClient(
            model=llm_config.get("model"),
            vertex_project=llm_config.get("vertex_project"),
            vertex_location=llm_config.get("vertex_location"),
            vertex_credentials=llm_config.get("vertex_credentials"),
            temperature=llm_config.get("temperature", 0.0),
            max_tokens=llm_config.get("max_tokens", 16384),
        )
        agent_config = self.config.get("agent", {})
        return BaseAgent(
            llm_client=client,
            tools=self._shared_tools,
            system_prompt=self._system_prompt,
            max_loops=agent_config.get("max_loops", 15),
            max_token_budget=agent_config.get("max_token_budget", 200000),
            verbose=self.verbose,
        )

    def _load_completed_qids(self) -> set:
        completed = set()
        if not self.predictions_file.exists():
            return completed
        with open(self.predictions_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    qid = data.get("qid") or data.get("id")
                    if qid and "pred_answer" in data:
                        completed.add(qid)
                except json.JSONDecodeError:
                    continue
        return completed

    def _append_prediction(self, prediction: Dict[str, Any]):
        with self.write_lock:
            with open(self.predictions_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    def _process_one(self, item: Dict[str, Any], agent: BaseAgent) -> Dict[str, Any]:
        qid = item.get("id") or item.get("qid")
        question = item["question"]
        gold_answer = item.get("answer", "")

        try:
            result = agent.run(question)
            return {
                "qid": qid,
                "question": question,
                "gold_answer": gold_answer,
                "pred_answer": result["answer"],
                "trajectory": result["trajectory"],
                "total_cost": result["total_cost"],
                "loops": result["loops"],
                "metadata": item.get("metadata", {}),
                **{k: v for k, v in result.items()
                   if k not in ("answer", "trajectory", "total_cost", "loops")},
            }
        except Exception as e:
            return {
                "qid": qid,
                "question": question,
                "gold_answer": gold_answer,
                "pred_answer": f"Error: {e}",
                "trajectory": [],
                "total_cost": 0,
                "loops": 0,
                "metadata": item.get("metadata", {}),
                "error": str(e),
            }

    def run(self):
        completed_qids = self._load_completed_qids()
        pending = [q for q in self.questions if (q.get("id") or q.get("qid")) not in completed_qids]

        print(f"Total: {len(self.questions)} | Completed: {len(completed_qids)} | Pending: {len(pending)}")

        if not pending:
            print("All questions completed!")
            return

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {}
            for item in pending:
                agent = self._create_agent()
                future = executor.submit(self._process_one, item, agent)
                futures[future] = item.get("id") or item.get("qid")

            with tqdm(total=len(pending), desc="Processing") as pbar:
                for future in as_completed(futures):
                    qid = futures[future]
                    try:
                        result = future.result()
                        self._append_prediction(result)
                    except Exception as e:
                        print(f"Error {qid}: {e}")
                    pbar.update(1)

        print(f"Results saved to: {self.predictions_file}")


def main():
    parser = argparse.ArgumentParser(description="HyProTwin A-RAG Batch Runner")
    parser.add_argument("--config", "-c", default="configs/hyprotwin.yaml")
    parser.add_argument("--output", "-o", default="results/hyprotwin/")
    parser.add_argument("--limit", "-l", type=int, default=None)
    parser.add_argument("--workers", "-w", type=int, default=5)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    runner = HyProTwinBatchRunner(
        config=config,
        output_dir=args.output,
        limit=args.limit,
        num_workers=args.workers,
        verbose=args.verbose,
    )
    runner.run()


if __name__ == "__main__":
    main()
```

**Step 4: Run test**

Run: `uv run pytest tests/test_batch_runner_hyprotwin.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/batch_runner_hyprotwin.py tests/test_batch_runner_hyprotwin.py
git commit -m "feat: add HyProTwin-adapted batch runner"
```

---

## Task 12: Qdrant Upload Script

**Files:**
- Create: `scripts/upload_to_qdrant.py`

**Step 1: Implement the upload script**

Create `scripts/upload_to_qdrant.py`:

```python
#!/usr/bin/env python3
"""Upload WSV chunks to Qdrant with embeddings and metadata payloads.

Usage:
    uv run python scripts/upload_to_qdrant.py \
        --config configs/hyprotwin.yaml
"""

import argparse
import json
from pathlib import Path

import litellm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm

from arag import Config


def main():
    parser = argparse.ArgumentParser(description="Upload chunks to Qdrant")
    parser.add_argument("--config", "-c", default="configs/hyprotwin.yaml")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--recreate", action="store_true", help="Recreate collection")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    data_config = config.get("data", {})
    qdrant_config = config.get("qdrant", {})
    embedding_config = config.get("embedding", {})
    llm_config = config.get("llm", {})

    # Load chunks
    chunks_file = data_config.get("chunks_file", "data/wsv/chunks.json")
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")

    # Connect to Qdrant
    client = QdrantClient(
        host=qdrant_config.get("host", "localhost"),
        port=qdrant_config.get("port", 6333),
    )
    collection = qdrant_config.get("collection", "hyprotwin")

    # Get embedding dimension from a test embedding
    model = embedding_config.get("model", "text-embedding-ada-002")
    vertex_kwargs = {}
    if llm_config.get("vertex_project"):
        vertex_kwargs["vertex_project"] = llm_config["vertex_project"]
    if llm_config.get("vertex_location"):
        vertex_kwargs["vertex_location"] = llm_config["vertex_location"]
    if llm_config.get("vertex_credentials"):
        vertex_kwargs["vertex_credentials"] = llm_config["vertex_credentials"]

    test_resp = litellm.embedding(model=model, input=["test"], **vertex_kwargs)
    dim = len(test_resp.data[0]["embedding"])
    print(f"Embedding model: {model}, dimension: {dim}")

    # Create/recreate collection
    if args.recreate:
        client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        print(f"Recreated collection '{collection}'")
    elif not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        print(f"Created collection '{collection}'")

    # Upload in batches
    for i in tqdm(range(0, len(chunks), args.batch_size), desc="Uploading"):
        batch = chunks[i : i + args.batch_size]
        texts = [c["text"] for c in batch]

        # Get embeddings
        resp = litellm.embedding(model=model, input=texts, **vertex_kwargs)
        vectors = [d["embedding"] for d in resp.data]

        # Build points
        points = []
        for j, (chunk, vector) in enumerate(zip(batch, vectors)):
            points.append(
                PointStruct(
                    id=i + j,
                    vector=vector,
                    payload={
                        "chunk_id": chunk["id"],
                        "text": chunk["text"],
                        "document_id": chunk.get("document_id", ""),
                        "structure_name": chunk.get("structure_name", ""),
                        "structure_id": chunk.get("structure_id", ""),
                        "inspection_year": chunk.get("inspection_year"),
                        "report_type": chunk.get("report_type", ""),
                        "doc_type": chunk.get("doc_type", ""),
                    },
                )
            )

        client.upsert(collection_name=collection, points=points)

    print(f"Uploaded {len(chunks)} chunks to '{collection}'")


if __name__ == "__main__":
    main()
```

**Step 2: Commit (don't run — requires Qdrant instance)**

```bash
git add scripts/upload_to_qdrant.py
git commit -m "feat: add Qdrant upload script for WSV chunks"
```

---

## Task 13: Extended Evaluation Script

**Files:**
- Create: `scripts/eval_hyprotwin.py`

**Step 1: Implement the extended evaluation**

Create `scripts/eval_hyprotwin.py`:

```python
#!/usr/bin/env python3
"""Extended evaluation for HyProTwin, adding document identification
and per-category metrics on top of A-RAG's standard eval.

Usage:
    uv run python scripts/eval_hyprotwin.py \
        --predictions results/hyprotwin/predictions.jsonl \
        --config configs/hyprotwin.yaml
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from arag import Config


def check_document_identification(trajectory: list, expected_docs: str) -> bool:
    """Check if the agent's trajectory accessed any of the expected documents."""
    if not expected_docs:
        return True

    expected = set()
    for doc in expected_docs.split(";"):
        doc = doc.strip()
        # Normalize: remove .pdf, lowercase
        base = doc.replace(".pdf", "").lower()
        expected.add(base)

    # Check tool calls for document references
    for entry in trajectory:
        tool_name = entry.get("tool_name", "")
        args = entry.get("arguments", {})
        result = entry.get("tool_result", "")

        # Check read_document calls
        if tool_name == "read_document":
            doc_id = args.get("document_id", "").lower()
            for exp in expected:
                if exp in doc_id or doc_id in exp:
                    return True

        # Check if result text mentions expected docs
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

    # Load predictions
    predictions = []
    with open(args.predictions, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))

    print(f"Loaded {len(predictions)} predictions")

    # Per-category stats
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

    # Print report
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

    # Save summary
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
```

**Step 2: Commit**

```bash
git add scripts/eval_hyprotwin.py
git commit -m "feat: add extended evaluation script with doc identification and per-category metrics"
```

---

## Task 14: End-to-End Smoke Test

**Files:**
- Create: `tests/test_e2e_hyprotwin.py`

**Step 1: Write the integration test**

Create `tests/test_e2e_hyprotwin.py`:

```python
"""End-to-end smoke test for HyProTwin A-RAG pipeline.

Tests the full pipeline without live API calls (mocked LLM).
Requires preprocessed data files from Task 5.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from arag import BaseAgent, ToolRegistry, Config
from arag.core.context import AgentContext
from arag.core.llm import LLMClient
from arag.tools.keyword_search import KeywordSearchTool
from arag.tools.read_document import ReadDocumentTool
from arag.tools.metadata_filter import MetadataFilterTool


DATA_DIR = Path("data/wsv")
CHUNKS_FILE = DATA_DIR / "chunks.json"
REGISTRY_FILE = DATA_DIR / "document_registry.json"


@pytest.fixture
def tools():
    """Set up tool registry with real data files."""
    if not CHUNKS_FILE.exists() or not REGISTRY_FILE.exists():
        pytest.skip("Preprocessed data not available. Run scripts/preprocess_wsv.py first.")

    registry = ToolRegistry()
    registry.register(KeywordSearchTool(chunks_file=str(CHUNKS_FILE)))
    registry.register(ReadDocumentTool(registry_file=str(REGISTRY_FILE)))
    registry.register(MetadataFilterTool(registry_file=str(REGISTRY_FILE)))
    return registry


def test_metadata_filter_finds_marktbreit(tools):
    """metadata_filter returns results for 'Marktbreit'."""
    ctx = AgentContext()
    result, log = tools.execute("metadata_filter", ctx, structure_name="Marktbreit")
    assert log["documents_found"] > 0
    assert "Marktbreit" in result


def test_keyword_search_finds_chunks(tools):
    """keyword_search returns results for a structure name."""
    ctx = AgentContext()
    result, log = tools.execute("keyword_search", ctx, keywords="Wehranlage Marktbreit")
    assert log.get("chunks_found", 0) > 0


def test_read_document_reads_file(tools):
    """read_document returns content for a valid document_id."""
    # First find a valid document_id via metadata_filter
    ctx = AgentContext()
    result, log = tools.execute("metadata_filter", ctx, structure_name="Marktbreit")
    assert log["documents_found"] > 0

    # Extract first document_id from result
    with open(REGISTRY_FILE) as f:
        registry = json.load(f)
    doc_id = next(
        (did for did in registry if "marktbreit" in registry[did].get("structure_name", "").lower()),
        None,
    )
    assert doc_id is not None

    result, log = tools.execute("read_document", ctx, document_id=doc_id)
    assert log["retrieved_tokens"] > 0
    assert len(result) > 100


def test_full_pipeline_mock_llm(tools):
    """Full agent run with mocked LLM responses."""
    # Simulate: LLM calls metadata_filter, then read_document, then answers

    # Find a real doc_id for the mock
    with open(REGISTRY_FILE) as f:
        registry = json.load(f)
    doc_id = next(iter(registry))

    call_count = 0

    def mock_chat(messages, tools=None, **kwargs):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            # First call: LLM calls metadata_filter
            return {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "tc_1",
                        "function": {
                            "name": "metadata_filter",
                            "arguments": json.dumps({"structure_name": "Marktbreit"}),
                        },
                    }],
                },
                "input_tokens": 100,
                "output_tokens": 50,
                "cost": 0.001,
            }
        elif call_count == 2:
            # Second call: LLM calls read_document
            return {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "tc_2",
                        "function": {
                            "name": "read_document",
                            "arguments": json.dumps({"document_id": doc_id}),
                        },
                    }],
                },
                "input_tokens": 500,
                "output_tokens": 50,
                "cost": 0.005,
            }
        else:
            # Third call: LLM gives final answer
            return {
                "message": {
                    "role": "assistant",
                    "content": "Die höchste Schadensklasse ist 3.",
                    "tool_calls": None,
                },
                "input_tokens": 2000,
                "output_tokens": 100,
                "cost": 0.01,
            }

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = mock_chat
    mock_llm.count_tokens.return_value = 10
    mock_llm.count_message_tokens.return_value = 100

    agent = BaseAgent(
        llm_client=mock_llm,
        tools=tools,
        system_prompt="Test prompt",
        max_loops=10,
        max_token_budget=200000,
    )

    result = agent.run("Welche Schäden gibt es an der Wehranlage Marktbreit?")

    assert result["answer"] == "Die höchste Schadensklasse ist 3."
    assert result["loops"] == 3
    assert len(result["trajectory"]) == 2  # Two tool calls
    assert result["trajectory"][0]["tool_name"] == "metadata_filter"
    assert result["trajectory"][1]["tool_name"] == "read_document"
```

**Step 2: Run the test**

Run: `uv run pytest tests/test_e2e_hyprotwin.py -v`
Expected: All PASS (after preprocessing data exists from Task 5)

**Step 3: Commit**

```bash
git add tests/test_e2e_hyprotwin.py
git commit -m "test: add end-to-end smoke tests for HyProTwin pipeline"
```

---

## Task 15: Run All Tests

**Step 1: Run the full test suite**

Run: `uv run pytest tests/ -v`
Expected: All PASS

**Step 2: Run linter**

Run: `uv run ruff check src/ scripts/ tests/`
Expected: Clean or minor fixable warnings

**Step 3: Fix any issues and commit**

```bash
git add -u
git commit -m "fix: address linter warnings"
```

---

## Execution Summary

| Task | Description | Depends on |
|------|-------------|------------|
| 1 | Add dependencies | — |
| 2 | LLM client (litellm/Vertex AI) | 1 |
| 3 | Metadata extraction | 1 |
| 4 | Chunking pipeline | 3 |
| 5 | Preprocessing script | 3, 4 |
| 6 | Eval catalog conversion | 1 |
| 7 | read_document tool | 1 |
| 8 | metadata_filter tool | 1 |
| 9 | semantic_search (Qdrant) | 1 |
| 10 | System prompt + config | — |
| 11 | Batch runner | 2, 7, 8, 10 |
| 12 | Qdrant upload script | 5, 9 |
| 13 | Extended eval script | 6 |
| 14 | E2E smoke test | 5, 7, 8, 11 |
| 15 | Run all tests | 1-14 |

**Parallelizable groups:**
- Tasks 3, 6, 7, 8, 9, 10 can all run in parallel after Task 1
- Tasks 4, 5 must be sequential after 3
- Tasks 11, 12, 13, 14 depend on earlier tasks
