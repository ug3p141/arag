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

        if context.is_chunk_read(read_key):
            return (
                f"Document '{document_id}' has already been read in this session.",
                {"retrieved_tokens": 0, "already_read": True},
            )

        entry = self._registry.get(document_id)
        if not entry:
            return (
                f"Document ID '{document_id}' not found in the registry. "
                f"Available documents can be found via metadata_filter or search tools.",
                {"retrieved_tokens": 0, "error": "not_found"},
            )

        source_file = Path(entry["source_file"])
        if not source_file.exists():
            return (
                f"Source file not found: {entry['source_file']}",
                {"retrieved_tokens": 0, "error": "file_not_found"},
            )

        text = source_file.read_text(encoding="utf-8")
        tokens = len(self._tokenizer.encode(text))

        context.mark_chunk_as_read(read_key)
        context.add_retrieval_log(
            tool_name="read_document",
            tokens=tokens,
            metadata={"document_id": document_id},
        )

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
