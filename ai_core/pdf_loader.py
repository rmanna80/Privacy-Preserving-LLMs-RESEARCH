from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .sensitive_extractors import SSN_PATTERN

logger = logging.getLogger(__name__)


def configure_tesseract() -> None:
    """
    Configure pytesseract from environment when provided.

    Expected environment variable:
    - TESSERACT_CMD
    """
    tesseract_cmd = os.getenv("TESSERACT_CMD")
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        logger.info("Configured Tesseract from TESSERACT_CMD.")


def _validate_pdf_dir(pdf_dir: str | Path) -> Path:
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"Directory not found: {pdf_dir}")
    if not pdf_dir.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {pdf_dir}")
    return pdf_dir


def _extract_page_text(page) -> str:
    return (page.get_text("text") or "").strip()


def _ocr_page(page, dpi: int = 300) -> str:
    pix = page.get_pixmap(dpi=dpi)
    image_mode = "RGB" if pix.alpha == 0 else "RGBA"
    image = Image.frombytes(image_mode, [pix.width, pix.height], pix.samples)
    return pytesseract.image_to_string(image).strip()


def _should_use_ocr(
    text: str,
    page_idx: int,
    ocr_strategy: str,
    ocr_first_n_pages: int,
) -> bool:
    """
    Decide whether OCR should run for a page.

    Supported strategies:
    - "never": never OCR
    - "always_first_n": OCR the first N pages regardless
    - "low_text": OCR when extracted text is very sparse
    - "ssn_fallback": OCR early pages when SSN-like content appears missing
    """
    if ocr_strategy == "never":
        return False

    if page_idx >= ocr_first_n_pages:
        return False

    if ocr_strategy == "always_first_n":
        return True

    if ocr_strategy == "low_text":
        return len(text.strip()) < 40

    if ocr_strategy == "ssn_fallback":
        ssn_count = len(SSN_PATTERN.findall(text))
        return ssn_count < 2

    return False


def load_pdfs(pdf_dir: str | Path) -> List[Document]:
    """
    Load all PDFs from a directory using direct PyMuPDF text extraction.
    """
    pdf_dir = _validate_pdf_dir(pdf_dir)
    docs: List[Document] = []

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        logger.info("Loading PDF: %s", pdf_path.name)
        pdf = fitz.open(str(pdf_path))
        try:
            for page_idx in range(len(pdf)):
                text = _extract_page_text(pdf[page_idx])
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(pdf_path),
                            "page": page_idx,
                            "extraction_method": "text",
                        },
                    )
                )
        finally:
            pdf.close()

    logger.info("Loaded %d raw pages from %s", len(docs), pdf_dir)
    return docs


def load_pdfs_hybrid(
    pdf_dir: str | Path,
    *,
    ocr_enabled: bool = True,
    ocr_strategy: str = "low_text",
    ocr_first_n_pages: int = 2,
    ocr_dpi: int = 300,
) -> List[Document]:
    """
    Hybrid PDF loading:
    - try direct text extraction first
    - optionally OCR selected pages as fallback

    Parameters
    ----------
    ocr_enabled:
        Whether OCR fallback is allowed.
    ocr_strategy:
        One of: "never", "always_first_n", "low_text", "ssn_fallback"
    ocr_first_n_pages:
        Only consider OCR for the first N pages.
    ocr_dpi:
        DPI used for page rasterization before OCR.
    """
    configure_tesseract()
    pdf_dir = _validate_pdf_dir(pdf_dir)
    docs: List[Document] = []

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        logger.info("Loading PDF: %s", pdf_path.name)
        pdf = fitz.open(str(pdf_path))
        try:
            for page_idx in range(len(pdf)):
                page = pdf[page_idx]
                text = _extract_page_text(page)
                extraction_method = "text"

                if ocr_enabled and _should_use_ocr(
                    text=text,
                    page_idx=page_idx,
                    ocr_strategy=ocr_strategy,
                    ocr_first_n_pages=ocr_first_n_pages,
                ):
                    try:
                        ocr_text = _ocr_page(page, dpi=ocr_dpi)
                        if len(ocr_text.strip()) > len(text.strip()):
                            text = ocr_text
                            extraction_method = "ocr"
                            logger.info(
                                "OCR used for %s page %s",
                                pdf_path.name,
                                page_idx,
                            )
                    except Exception as exc:
                        logger.warning(
                            "OCR failed for %s page %s: %s",
                            pdf_path.name,
                            page_idx,
                            exc,
                        )

                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(pdf_path),
                            "page": page_idx,
                            "extraction_method": extraction_method,
                        },
                    )
                )
        finally:
            pdf.close()

    logger.info("Loaded %d pages from %s using hybrid extraction", len(docs), pdf_dir)
    return docs


def split_documents(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    """
    Split documents into overlapping chunks for retrieval.
    """
    if not docs:
        logger.info("No documents to split.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)

    logger.info(
        "Split %d docs into %d chunks (chunk_size=%d, overlap=%d)",
        len(docs),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks