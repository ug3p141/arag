"""Tests for Qdrant-based semantic search tool."""

from unittest.mock import MagicMock
from arag.core.context import AgentContext
from arag.tools.semantic_search_qdrant import QdrantSemanticSearchTool


def _make_mock_qdrant():
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
    client = _make_mock_qdrant()
    embed_fn = MagicMock(return_value=[0.1] * 384)
    tool = QdrantSemanticSearchTool(
        qdrant_client=client, collection_name="test", embedding_fn=embed_fn,
    )
    ctx = AgentContext()
    result, log = tool.execute(ctx, query="Korrosion Walze", top_k=5)
    assert "6120_2009_SBL_TR" in result
    assert "Korrosion" in result
    assert log["chunks_found"] == 1
    client.search.assert_called_once()


def test_semantic_search_empty_results():
    client = MagicMock()
    client.search.return_value = []
    embed_fn = MagicMock(return_value=[0.1] * 384)
    tool = QdrantSemanticSearchTool(
        qdrant_client=client, collection_name="test", embedding_fn=embed_fn,
    )
    ctx = AgentContext()
    result, log = tool.execute(ctx, query="nothing here")
    assert log["chunks_found"] == 0


def test_semantic_search_schema():
    tool = QdrantSemanticSearchTool.__new__(QdrantSemanticSearchTool)
    schema = tool.get_schema()
    params = schema["function"]["parameters"]["properties"]
    assert "query" in params
    assert "top_k" in params


def test_tool_name():
    tool = QdrantSemanticSearchTool.__new__(QdrantSemanticSearchTool)
    assert tool.name == "semantic_search"
