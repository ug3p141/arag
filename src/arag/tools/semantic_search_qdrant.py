"""Semantic search via Qdrant vector database."""

from typing import Any, Callable, Dict, List, Tuple

import tiktoken

from arag.tools.base import BaseTool
from arag.core.context import AgentContext


class QdrantSemanticSearchTool(BaseTool):
    """Semantic search over inspection report chunks using Qdrant."""

    def __init__(self, qdrant_client, collection_name: str, embedding_fn: Callable[[str], List[float]]):
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
            tokens=total_tokens,
            metadata={"query": query, "chunks_found": len(results)},
        )

        return (
            f"Found {len(results)} matching chunks:\n\n" + "\n".join(lines),
            {"chunks_found": len(results), "retrieved_tokens": total_tokens},
        )
