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
            tokens=0,
            metadata={"documents_found": len(matches), "filters": kwargs},
        )

        return (
            "\n".join(lines),
            {"documents_found": len(matches)},
        )
