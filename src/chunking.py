import logging
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.chunking import PDFChunker

PDF_PATH = ROOT_DIR / "supporting_files" / "History_of_the_FIFA_World_Cup.pdf"
OUTPUT_PATH = ROOT_DIR / "supporting_files" / "generated_chunks.json"


def get_parameters() -> dict[str, Any]:
    """Return the default chunking configuration.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the PDF path, output path, chunk size, and overlap ratio.
    """
    return {
        "pdf_path": PDF_PATH,
        "output_path": OUTPUT_PATH,
        "chunk_size": 600,
        "overlap_ratio": 0.1,
    }


def main(params: dict[str, Any] | None = None) -> None:
    """Run the PDF chunking workflow.

    Parameters
    ----------
    params : dict[str, Any] or None, optional
        Parameters used to configure the chunking workflow. If not provided,
        the default values are used.
    """
    config = params if params is not None else get_parameters()
    pdf_path = Path(config["pdf_path"])
    output_path = Path(config["output_path"])

    chunker = PDFChunker(
        chunk_size=config["chunk_size"],
        overlap_ratio=config["overlap_ratio"],
    )

    chunker.process(
        pdf_path=pdf_path,
        output_path=output_path,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()