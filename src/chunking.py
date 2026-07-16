import argparse
import logging
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.chunking_builder import PDFChunker

PDF_PATH = ROOT_DIR / "supporting_files" / "History_of_the_FIFA_World_Cup.pdf"
OUTPUT_PATH = ROOT_DIR / "supporting_files" / "generated_chunks.json"


def get_parameters(chunk_size: int = 600, overlap_ratio: float = 0.1) -> dict[str, Any]:
    """Return the default chunking configuration.

    Parameters
    ----------
    chunk_size : int, optional
        Number of words to include per chunk.
    overlap_ratio : float, optional
        Fraction of overlap between adjacent chunks.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the PDF path, output path, chunk size, and overlap ratio.
    """
    return {
        "pdf_path": PDF_PATH,
        "output_path": OUTPUT_PATH,
        "chunk_size": chunk_size,
        "overlap_ratio": overlap_ratio,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for chunking."""
    parser = argparse.ArgumentParser(
        description="Chunk a PDF into overlapping text blocks"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        required=False,
        default=600,
        help="Number of words per chunk",
    )
    parser.add_argument(
        "--overlap-ratio",
        type=float,
        required=False,
        default=0.1,
        help="Overlap ratio between chunks",
    )
    return parser.parse_args()


def main(params: dict[str, Any] | None = None) -> None:
    """Run the PDF chunking workflow.

    Parameters
    ----------
    params : dict[str, Any] or None, optional
        Parameters used to configure the chunking workflow. If not provided,
        the default values are used.
    """
    if params is None:
        args = parse_args()
        config = get_parameters(
            chunk_size=args.chunk_size, overlap_ratio=args.overlap_ratio
        )
    else:
        config = params

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
