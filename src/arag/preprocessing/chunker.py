"""Structure-aware chunking for WSV inspection reports."""

import re
from typing import Dict, List, Optional


def extract_inspection_year(text: str, report_date: str = None) -> Optional[int]:
    """Extract inspection year from document text, falling back to report date."""
    for pattern in [
        r"Inspektionsjahr[:\s]*(\d{4})",
        r"Prüfjahr[:\s]*(\d{4})",
    ]:
        m = re.search(pattern, text)
        if m:
            return int(m.group(1))
    if report_date:
        return int(report_date[:4])
    return None


def _build_header(document_id: str, metadata: Dict) -> str:
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
    sections = []
    current_title = "header"
    current_lines = []

    for line in text.split("\n"):
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
    header_prefix = _build_header(document_id, metadata)
    sections = _split_into_sections(text)
    chunks = []
    seq = 0
    max_chars = max_chunk_tokens * 4

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
