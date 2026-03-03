"""Tests for metadata extraction from WSV filenames."""

from arag.preprocessing.metadata import extract_metadata_from_filename, STRUCTURE_LOOKUP


def test_bericht_standard_pattern():
    meta = extract_metadata_from_filename(
        "Bericht_6120_SBL_TR_18.02.2010_complete.md",
        directory_name="30_Wehranlage_Marktbreit_213-6326002",
    )
    assert meta["structure_id"] == "6120"
    assert meta["report_type"] == "SBL_TR"
    assert meta["report_date"] == "2010-02-18"
    assert meta["structure_name"] == "Wehranlage Marktbreit"
    assert meta["doc_type"] == "bericht"


def test_bericht_sbl2_pattern():
    meta = extract_metadata_from_filename(
        "Bericht_6120_SBL2_SR_04.01.2022_complete.md",
        directory_name="30_Wehranlage_Marktbreit_213-6326002",
    )
    assert meta["structure_id"] == "6120"
    assert meta["report_type"] == "SBL2_SR"
    assert meta["report_date"] == "2022-01-04"


def test_bericht_with_extra_segment():
    meta = extract_metadata_from_filename(
        "Bericht_6120_2-33_TR_05.01.2009_complete.md",
        directory_name="30_Wehranlage_Marktbreit_213-6326002",
    )
    assert meta["structure_id"] == "6120"
    assert meta["report_date"] == "2009-01-05"


def test_besichtigungsbericht_pattern():
    meta = extract_metadata_from_filename(
        "2014_12_12_Besichtigungsbericht_2136326002_complete.md",
        directory_name="30_Wehranlage_Marktbreit_213-6326002",
    )
    assert meta["report_date"] == "2014-12-12"
    assert meta["object_id"] == "2136326002"
    assert meta["doc_type"] == "besichtigungsbericht"
    assert meta["structure_name"] == "Wehranlage Marktbreit"


def test_numeric_code_pattern():
    meta = extract_metadata_from_filename(
        "612-0000884-0000_complete.md",
        directory_name="30_Wehranlage_Marktbreit_213-6326002",
    )
    assert meta["doc_type"] == "archive"
    assert meta["archive_code"] == "612-0000884-0000"
    assert meta["structure_name"] == "Wehranlage Marktbreit"


def test_baw_report_pattern():
    meta = extract_metadata_from_filename(
        "95100_BAW_JHWSVPR_Gesamt_Bericht_complete.md",
        directory_name="30_Wehranlage_Marktbreit_213-6326002",
    )
    assert meta["doc_type"] == "baw_report"
    assert meta["structure_name"] == "Wehranlage Marktbreit"


def test_altenrheine_structure():
    meta = extract_metadata_from_filename(
        "Bericht_4140_SBL2_BU_20.01.2010_complete.md",
        directory_name="39_Schiffsschleusenanlage_Altenrheine_311-3710006",
    )
    assert meta["structure_id"] == "4140"
    assert meta["structure_name"] == "Schiffsschleusenanlage Altenrheine"


def test_structure_lookup_table():
    assert "Wehranlage Marktbreit" in STRUCTURE_LOOKUP.values()
    assert "Schiffsschleusenanlage Altenrheine" in STRUCTURE_LOOKUP.values()


def test_filename_with_space_suffix():
    meta = extract_metadata_from_filename(
        "Bericht_6120_SBL2_GM_26.01.2010 2_complete.md",
        directory_name="30_Wehranlage_Marktbreit_213-6326002",
    )
    assert meta["structure_id"] == "6120"
    assert meta["report_type"] == "SBL2_GM"


def test_altenrheine_fgb_pattern():
    meta = extract_metadata_from_filename(
        "Bericht_81200_FGB_BR_28.12.2023_complete.md",
        directory_name="39_Schiffsschleusenanlage_Altenrheine_311-3710006",
    )
    assert meta["structure_id"] == "81200"
    assert meta["report_type"] == "FGB_BR"
    assert meta["report_date"] == "2023-12-28"


def test_baw_report_with_space_suffix():
    meta = extract_metadata_from_filename(
        "95100_BAW_JHWSVPR_Gesamt_Bericht 5_complete.md",
        directory_name="30_Wehranlage_Marktbreit_213-6326002",
    )
    assert meta["doc_type"] == "baw_report"


def test_archive_999_pattern():
    meta = extract_metadata_from_filename(
        "999-0241665-0001_complete.md",
        directory_name="30_Wehranlage_Marktbreit_213-6326002",
    )
    assert meta["doc_type"] == "archive"
    assert meta["archive_code"] == "999-0241665-0001"


def test_archive_414_pattern():
    meta = extract_metadata_from_filename(
        "414-0017398-0000_complete.md",
        directory_name="39_Schiffsschleusenanlage_Altenrheine_311-3710006",
    )
    assert meta["doc_type"] == "archive"
    assert meta["archive_code"] == "414-0017398-0000"
    assert meta["structure_name"] == "Schiffsschleusenanlage Altenrheine"
