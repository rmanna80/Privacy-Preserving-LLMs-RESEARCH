# ai_core/pdf_loader.py

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdfs(pdf_dir: str | Path) -> List[Document]:
    """
    Load all PDFs from a directory into LangChain Documents using PyPDFLoader.
    """
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"Directory not found: {pdf_dir}")

    docs: List[Document] = []

    for pdf_path in pdf_dir.glob("*.pdf"):
        print(f"Loading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        file_docs = loader.load()
        # Keep track of where each piece came from
        for d in file_docs:
            d.metadata.setdefault("source", str(pdf_path))
        docs.extend(file_docs)

    print(f"Loaded {len(docs)} raw pages from {pdf_dir}")
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
