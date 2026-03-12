# ai_core/pdf_loader.py

from __future__ import annotations

from pathlib import Path
from typing import List

from .sensitive_extractors import SSN_PATTERN
from PIL import Image
import pytesseract


import sys

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure Tesseract path for Windows
if sys.platform == "win32":
    # Update this path if you installed Tesseract somewhere else
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
print(pytesseract.get_tesseract_version())

def load_pdfs(pdf_dir: str | Path) -> List[Document]:
    """
    Load all PDFs from a directory into LangChain Documents using PyMuPDF.
    This usually extracts more reliably from IRS-style forms than pypdf.
    """
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"Directory not found: {pdf_dir}")

    docs: List[Document] = []

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        print(f"Loading: {pdf_path.name}")
        pdf = fitz.open(str(pdf_path))
        try:
            for page_idx in range(len(pdf)):
                text = pdf[page_idx].get_text("text") or ""
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": str(pdf_path), "page": page_idx},
                    )
                )
        finally:
            pdf.close()

    print(f"Loaded {len(docs)} raw pages from {pdf_dir}")
    return docs


def load_pdfs_hybrid(pdf_dir: str | Path) -> List[Document]:
    """
    Try text extraction first, use OCR as fallback if SSN pattern not found.
    """
    pdf_dir = Path(pdf_dir)
    docs: List[Document] = []

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        print(f"Loading: {pdf_path.name}")
        pdf = fitz.open(str(pdf_path))
        
        for page_idx in range(len(pdf)):
            page = pdf[page_idx]
            
            # Try normal text extraction first
            text = page.get_text("text") or ""
            
            # Check if we got SSNs
            ssn_count = len(SSN_PATTERN.findall(text))
            
            # If we found fewer than 2 SSNs on page 0, try OCR
            if page_idx == 0 and ssn_count < 2:
                print(f"  Page {page_idx}: Only found {ssn_count} SSN(s), trying OCR...")
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = pytesseract.image_to_string(img)
                ssn_count_ocr = len(SSN_PATTERN.findall(text))
                print(f"  OCR found {ssn_count_ocr} SSN(s)")
                
                # ADD THIS DEBUG - Check for the Smith document
                if "Smith" in pdf_path.name:  # Changed from "John_and_Sally"
                    print("\n" + "="*80)
                    print(f"[DEBUG OCR OUTPUT for {pdf_path.name}]")
                    print("="*80)
                    print("First 2000 characters of OCR text:")
                    print(text[:2000])
                    print("\n" + "="*80)
                    print("[DEBUG] Searching for '111' in OCR text:", '111' in text)
                    print("[DEBUG] Searching for '222' in OCR text:", '222' in text)
                    print("[DEBUG] Searching for 'JOHN' in OCR text:", 'JOHN' in text)
                    print("[DEBUG] Searching for 'SALLY' in OCR text:", 'SALLY' in text)
                    
                    # Show context around SALLY
                    if 'SALLY' in text.upper():
                        idx = text.upper().find('SALLY')
                        print(f"\n[DEBUG] Text around SALLY in OCR:")
                        print(text[max(0, idx-100):idx+300])
                    print("="*80 + "\n")
            
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": str(pdf_path), "page": page_idx}
                )
            )
        
        pdf.close()

    return docs


def split_documents(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    """
    Split documents into overlapping chunks suitable for retrieval later.
    """
    if not docs:
        print("No documents to split.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    print(
        f"Split {len(docs)} docs into {len(chunks)} chunks "
        f"(chunk_size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks