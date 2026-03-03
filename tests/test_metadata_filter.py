"""Tests for metadata_filter tool."""

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
    tool = MetadataFilterTool(registry_file=str(_make_registry(tmp_path)))
    ctx = AgentContext()
    result, log = tool.execute(ctx, structure_name="Marktbreit")
    assert "6120_2009_SBL_TR" in result
    assert "6120_2018_SBL2_SR" in result
    assert "4140" not in result


def test_filter_by_year(tmp_path):
    tool = MetadataFilterTool(registry_file=str(_make_registry(tmp_path)))
    ctx = AgentContext()
    result, log = tool.execute(ctx, inspection_year=2009)
    assert "6120_2009_SBL_TR" in result
    assert "2018" not in result


def test_filter_combined(tmp_path):
    tool = MetadataFilterTool(registry_file=str(_make_registry(tmp_path)))
    ctx = AgentContext()
    result, log = tool.execute(ctx, structure_name="Marktbreit", inspection_year=2018)
    assert "6120_2018_SBL2_SR" in result
    assert "6120_2009" not in result


def test_filter_no_match(tmp_path):
    tool = MetadataFilterTool(registry_file=str(_make_registry(tmp_path)))
    ctx = AgentContext()
    result, log = tool.execute(ctx, structure_name="Nonexistent")
    assert "no documents" in result.lower() or log["documents_found"] == 0


def test_filter_no_params(tmp_path):
    tool = MetadataFilterTool(registry_file=str(_make_registry(tmp_path)))
    ctx = AgentContext()
    result, log = tool.execute(ctx)
    assert "at least one" in result.lower()


def test_tool_schema(tmp_path):
    tool = MetadataFilterTool(registry_file=str(_make_registry(tmp_path)))
    schema = tool.get_schema()
    params = schema["function"]["parameters"]["properties"]
    assert "structure_name" in params
    assert "structure_id" in params
    assert "inspection_year" in params
    assert "report_type" in params
