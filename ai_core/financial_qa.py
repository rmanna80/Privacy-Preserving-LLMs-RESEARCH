# ai_core/financial_qa.py

from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document

from .pdf_loader import load_pdfs, split_documents


class FinancialQASystem:
    """
    Stage 1 version:
    - Loads PDFs from a directory
    - Splits them into chunks
    """

    def __init__(
        self,
        docs_dir: str | Path,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        verbose: bool = True,
    ) -> None:
        self.docs_dir = Path(docs_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.verbose = verbose

        self.raw_docs: List[Document] = []
        self.chunks: List[Document] = []

    def index_documents(self) -> None:
        """
        Load PDFs and split them into chunks.
        """
        if self.verbose:
            print("Loading PDFs...")
        self.raw_docs = load_pdfs(self.docs_dir)

        if self.verbose:
            print("Splitting documents into chunks...")
        self.chunks = split_documents(
            self.raw_docs,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        if self.verbose:
            print("Indexing complete.")
            print(f"Raw docs loaded: {len(self.raw_docs)}")
            print(f"Chunks created: {len(self.chunks)}")

    def preview_chunk(self, idx: int = 0, length: int = 400) -> None:
        """
        Print a preview of one document chunk for debugging.
        """
        if not self.chunks:
            print("No chunks available. Call index_documents() first.")
            return

        if idx < 0 or idx >= len(self.chunks):
            print(f"Index {idx} out of range.")
            return

        chunk = self.chunks[idx]
        print("----- Chunk Preview -----")
        print(chunk.page_content[:length])
        print("\n[metadata]:", chunk.metadata)
        print("-------------------------")
