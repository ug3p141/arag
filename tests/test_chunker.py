"""Tests for structure-aware chunking of WSV inspection reports."""

from arag.preprocessing.chunker import chunk_document, extract_inspection_year


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
    chunks = chunk_document(
        text=SAMPLE_DOC,
        document_id="6120_2009_SBL_TR",
        metadata={"structure_name": "Wehranlage Marktbreit", "inspection_year": 2009},
    )
    assert isinstance(chunks, list)
    assert len(chunks) > 0


def test_chunk_has_required_fields():
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
    chunks = chunk_document(
        text=SAMPLE_DOC,
        document_id="6120_2009_SBL_TR",
        metadata={"structure_name": "Wehranlage Marktbreit", "inspection_year": 2009},
    )
    for chunk in chunks:
        assert chunk["text"].startswith("[")


def test_kopfdaten_chunk_is_first():
    chunks = chunk_document(
        text=SAMPLE_DOC,
        document_id="6120_2009_SBL_TR",
        metadata={"structure_name": "Wehranlage Marktbreit"},
    )
    assert "Objektidentnr" in chunks[0]["text"] or "Prüfbericht" in chunks[0]["text"]


def test_inspection_year_extracted():
    year = extract_inspection_year(SAMPLE_DOC, report_date="2010-02-18")
    assert year == 2009


def test_inspection_year_fallback_to_date():
    year = extract_inspection_year("No year mentioned here.", report_date="2018-12-19")
    assert year == 2018


def test_chunk_metadata_propagated():
    chunks = chunk_document(
        text=SAMPLE_DOC,
        document_id="6120_2009_SBL_TR",
        metadata={"structure_name": "Wehranlage Marktbreit", "structure_id": "6120"},
    )
    for chunk in chunks:
        assert chunk["document_id"] == "6120_2009_SBL_TR"
        assert chunk["structure_name"] == "Wehranlage Marktbreit"


def test_large_section_gets_split():
    """A section larger than max_chunk_tokens gets split at paragraph boundaries."""
    big_text = "## Page 1\n\nHeader info.\n\n## Page 2\n\n" + "\n\n".join(
        [f"Paragraph {i} with enough text to make it count. " * 20 for i in range(30)]
    )
    chunks = chunk_document(
        text=big_text,
        document_id="test",
        metadata={},
        max_chunk_tokens=500,
    )
    assert len(chunks) > 2  # Should be split into multiple chunks
