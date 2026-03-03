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
    r"^95100_BAW_.+?(?:\s+\d+)?_complete\.md$"
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
    parts = directory_name.split("_", 1)
    if len(parts) > 1:
        return parts[1].rsplit("_", 1)[0].replace("_", " ")
    return directory_name


def extract_metadata_from_filename(
    filename: str,
    directory_name: str = "",
) -> Dict[str, Optional[str]]:
    """Extract metadata from a WSV document filename."""
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
