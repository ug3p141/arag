"""Tests for read_document tool."""

import json
from pathlib import Path

from arag.core.context import AgentContext
from arag.tools.read_document import ReadDocumentTool


def _make_registry(tmp_path: Path) -> tuple[Path, Path]:
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
    registry_file, _ = _make_registry(tmp_path)
    tool = ReadDocumentTool(registry_file=str(registry_file))
    ctx = AgentContext()
    result, log = tool.execute(ctx, document_id="6120_2009_SBL_TR")
    assert "Test Document" in result
    assert "Some content about damages" in result
    assert log["retrieved_tokens"] > 0


def test_read_document_unknown_id(tmp_path):
    registry_file, _ = _make_registry(tmp_path)
    tool = ReadDocumentTool(registry_file=str(registry_file))
    ctx = AgentContext()
    result, log = tool.execute(ctx, document_id="nonexistent")
    assert "not found" in result.lower()


def test_read_document_marks_read(tmp_path):
    registry_file, _ = _make_registry(tmp_path)
    tool = ReadDocumentTool(registry_file=str(registry_file))
    ctx = AgentContext()
    tool.execute(ctx, document_id="6120_2009_SBL_TR")
    assert ctx.is_chunk_read("doc:6120_2009_SBL_TR")


def test_read_document_skips_reread(tmp_path):
    registry_file, _ = _make_registry(tmp_path)
    tool = ReadDocumentTool(registry_file=str(registry_file))
    ctx = AgentContext()
    tool.execute(ctx, document_id="6120_2009_SBL_TR")
    result2, log2 = tool.execute(ctx, document_id="6120_2009_SBL_TR")
    assert "already been read" in result2.lower()
    assert log2["retrieved_tokens"] == 0


def test_tool_name_and_schema():
    tool = ReadDocumentTool.__new__(ReadDocumentTool)
    assert tool.name == "read_document"
    # Can't call get_schema without init, just check name
