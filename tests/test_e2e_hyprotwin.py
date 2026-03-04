"""End-to-end smoke test for HyProTwin A-RAG pipeline.

Tests the full pipeline without live API calls (mocked LLM).
Requires preprocessed data files.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from arag import BaseAgent, ToolRegistry
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
    if not CHUNKS_FILE.exists() or not REGISTRY_FILE.exists():
        pytest.skip("Preprocessed data not available. Run scripts/preprocess_wsv.py first.")
    registry = ToolRegistry()
    registry.register(KeywordSearchTool(chunks_file=str(CHUNKS_FILE)))
    registry.register(ReadDocumentTool(registry_file=str(REGISTRY_FILE)))
    registry.register(MetadataFilterTool(registry_file=str(REGISTRY_FILE)))
    return registry


def test_metadata_filter_finds_marktbreit(tools):
    ctx = AgentContext()
    result, log = tools.execute("metadata_filter", ctx, structure_name="Marktbreit")
    assert log["documents_found"] > 0
    assert "Marktbreit" in result


def test_keyword_search_finds_chunks(tools):
    ctx = AgentContext()
    result, log = tools.execute(
        "keyword_search", ctx, keywords=["Wehranlage", "Marktbreit"]
    )
    assert log.get("chunks_found", 0) > 0


def test_read_document_reads_file(tools):
    ctx = AgentContext()
    # Get a valid doc_id from registry
    with open(REGISTRY_FILE) as f:
        registry = json.load(f)
    doc_id = next(
        (
            did
            for did in registry
            if "marktbreit" in (registry[did].get("structure_name") or "").lower()
        ),
        None,
    )
    assert doc_id is not None

    # Check source file exists before attempting read
    source_file = Path(registry[doc_id]["source_file"])
    if not source_file.exists():
        pytest.skip(f"Source file not available: {source_file}")

    result, log = tools.execute("read_document", ctx, document_id=doc_id)
    assert log["retrieved_tokens"] > 0
    assert len(result) > 100


def test_full_pipeline_mock_llm(tools):
    """Full agent run with mocked LLM responses."""
    # Find a real doc_id for Marktbreit
    with open(REGISTRY_FILE) as f:
        registry = json.load(f)
    doc_id = next(
        (
            did
            for did in registry
            if "marktbreit" in (registry[did].get("structure_name") or "").lower()
        ),
        None,
    )
    assert doc_id is not None

    # Check source file exists (read_document needs it)
    source_file = Path(registry[doc_id]["source_file"])
    if not source_file.exists():
        pytest.skip(f"Source file not available: {source_file}")

    call_count = 0

    def mock_chat(messages, tools=None, **kwargs):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            return {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc_1",
                            "function": {
                                "name": "metadata_filter",
                                "arguments": json.dumps(
                                    {"structure_name": "Marktbreit"}
                                ),
                            },
                        }
                    ],
                },
                "input_tokens": 100,
                "output_tokens": 50,
                "cost": 0.001,
            }
        elif call_count == 2:
            return {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc_2",
                            "function": {
                                "name": "read_document",
                                "arguments": json.dumps({"document_id": doc_id}),
                            },
                        }
                    ],
                },
                "input_tokens": 500,
                "output_tokens": 50,
                "cost": 0.005,
            }
        else:
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
    assert len(result["trajectory"]) == 2
    assert result["trajectory"][0]["tool_name"] == "metadata_filter"
    assert result["trajectory"][1]["tool_name"] == "read_document"
