"""Static helpers for locating chunk and index files inside a declared folder."""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT_DIR / "supporting_files"


def get_chunks_file(folder: str | Path = DEFAULT_DATA_DIR) -> Path:
    """Return the single JSON chunks file found in the given folder.

    Args:
        folder: Directory to search for a chunks JSON file.

    Returns:
        Path to the JSON file found in the folder.
    """
    folder = Path(folder)
    json_files = sorted(folder.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON chunks file found in: {folder}")
    if len(json_files) > 1:
        raise ValueError(f"Multiple JSON chunk files found in {folder}: {json_files}")

    return json_files[0]


def get_index_file(
    folder: str | Path = DEFAULT_DATA_DIR,
    default_name: str = "index.faiss",
) -> Path:
    """Return the single FAISS index file found in the given folder.

    If no ``.faiss`` file exists yet (e.g. before the first build), a path
    for ``default_name`` inside the folder is returned instead of raising.

    Args:
        folder: Directory to search for a FAISS index file.
        default_name: Filename to use when no index file exists yet.

    Returns:
        Path to the existing (or not-yet-created) FAISS index file.
    """
    folder = Path(folder)
    faiss_files = sorted(folder.glob("*.faiss"))

    if len(faiss_files) > 1:
        raise ValueError(f"Multiple FAISS index files found in {folder}: {faiss_files}")

    if faiss_files:
        return faiss_files[0]

    return folder / default_name


def get_pdf_file(folder: str | Path = DEFAULT_DATA_DIR) -> Path:
    """Return the single PDF file found in the given folder.

    Args:
        folder: Directory to search for a PDF file.

    Returns:
        Path to the PDF file found in the folder.
    """
    folder = Path(folder)
    pdf_files = sorted(folder.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF file found in: {folder}")
    if len(pdf_files) > 1:
        raise ValueError(f"Multiple PDF files found in {folder}: {pdf_files}")

    return pdf_files[0]
