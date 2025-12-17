# ai_core/vector_store.py

from pathlib import Path
from typing import Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import shutil


def build_or_load_vectorstore(
    chunks: Sequence[Document],
    persist_dir: Path,
    embeddings: Embeddings,
    force_rebuild: bool = False,
) -> Chroma:
    """
    Build or load a Chroma vector store for the document chunks.

    - If force_rebuild=True, delete existing DB and rebuild.
    - If a DB exists and force_rebuild=False, load it instead.
    """

    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    db_files_exist = any(persist_dir.glob("*.sqlite3")) or any(
        persist_dir.glob("chroma-*")
    )

    if db_files_exist and not force_rebuild:
        print(f"Loading existing Chroma DB from {persist_dir}")
        return Chroma(
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )

    if not chunks:
        raise ValueError("No document chunks provided to create Chroma DB.")

    if db_files_exist and force_rebuild:
        print(f"Removing existing Chroma DB at {persist_dir} (force_rebuild=True)")
        shutil.rmtree(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating new Chroma DB at {persist_dir}")
    db = Chroma.from_documents(
        documents=list(chunks),
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )

    print("Chroma DB created and persisted.")
    return db
