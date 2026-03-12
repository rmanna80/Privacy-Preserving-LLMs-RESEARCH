# pdf_loader_pypdf.py

from __future__ import annotations
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from pypdf import PdfReader

def load_pdfs_pypdf(pdf_dir: str | Path) -> List[Document]:
    pdf_dir = Path(pdf_dir)
    docs: List[Document] = []

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": str(pdf_path), "page": i},
                )
            )
    return docs
