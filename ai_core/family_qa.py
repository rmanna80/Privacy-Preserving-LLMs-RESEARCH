"""
ai_core/family_qa.py — Family-scoped RAG system.

Sub-phase 4c: wrap the existing FinancialQASystem to index and query
documents scoped to a specific Family rather than a client folder.

How it differs from FinancialQASystem
-------------------------------------
FinancialQASystem reads PDFs from a single docs_dir (one folder).
FamilyQASystem reads from the Document table in wealth.db, grabs each
file_path, and feeds them to the same underlying pipeline. All retrieval
gets filtered by family_id so chats only see that family's documents.

Indexing strategy
-----------------
For 4c we rebuild the family's Chroma collection on each upload. Simple,
reliable, but slow for large libraries. Optimization (incremental
indexing) deferred to a later phase.

Each family gets its own Chroma collection at:
    data/chroma/family_<id>/

Why per-family collections instead of one collection with a family_id
filter: it's the simpler model. We never have cross-family leakage by
construction. The cost is a bit more disk usage; for a single-tenant
deployment that's fine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

# from langchain_core.documents import Document as LangchainDoc

from ai_core.financial_qa import FinancialQASystem


def family_vectorstore_dir(family_id: int) -> Path:
    """Where this family's Chroma collection lives."""
    p = Path("data/chroma") / f"family_{family_id}"
    p.mkdir(parents=True, exist_ok=True)
    return p

def _release_chroma_handles() -> None:
    """Best-effort release of chromadb's cached clients so Windows will
    allow deleting a collection directory. Chroma caches client systems
    per path; without clearing, rmtree hits locked .bin files."""
    import gc
    try:
        from chromadb.api.client import SharedSystemClient
        SharedSystemClient.clear_system_cache()
    except Exception:
        pass
    gc.collect()


class FamilyQASystem(FinancialQASystem):
    """A FinancialQASystem that indexes a specific family's documents.

    Constructed from a family_id rather than a docs_dir. It pulls the
    list of documents from wealth.db and feeds them to the parent class's
    indexing pipeline.

    The parent class still handles all the chat / retrieval / privacy
    logic — we just override how documents are discovered.
    """

    def __init__(self, family_id: int, verbose: bool = False) -> None:
        # We point the parent at a per-family Chroma directory but use
        # a dummy docs_dir — we override index_documents to load from DB
        # instead of scanning a folder.
        dummy_docs_dir = Path("data/chroma") / f"family_{family_id}_input"
        dummy_docs_dir.mkdir(parents=True, exist_ok=True)

        super().__init__(
            docs_dir=dummy_docs_dir,
            db_dir=str(family_vectorstore_dir(family_id)),
            chunk_size=1200,
            chunk_overlap=200,
            verbose=verbose,
        )
        self.family_id = family_id

    def index_documents(self, force_rebuild: bool = False) -> None:
        """Load all non-archived documents for this family and index them.

        Strategy: copy each of this family's PDFs into a single staging
        directory, then use the existing load_pdfs_hybrid() pipeline.
        That keeps all the OCR-fallback logic in one place and avoids
        depending on functions that don't exist.
        """
        import shutil

        # Lazy imports to avoid circular dependencies
        from db.repositories import list_documents_for_family
        from ai_core.pdf_loader import load_pdfs_hybrid, split_documents
        from ai_core.embeddings import LocalEmbeddings
        from ai_core.vector_store import build_or_load_vectorstore
        from ai_core.llm_ollama import build_ollama_llm
        from ai_core.qa_chain import build_qa_chain

        if self.verbose:
            print(f"[FamilyQA] Indexing documents for family_id={self.family_id}")

        # 1) Pull doc metadata from the DB
        docs_meta = list_documents_for_family(self.family_id)

        self.embeddings = LocalEmbeddings()
        self.llm = build_ollama_llm()

        if not docs_meta:
            if self.verbose:
                print(f"[FamilyQA] No documents to index for family_id={self.family_id}")
            self.raw_docs = []
            self.chunks = []
            self.vector_store = None
            self.retriever = None
            self.qa_chain = None
            return

        # 2) Stage all the family's PDFs in a clean directory so the
        #    existing pipeline can scan them. Clean it first to handle
        #    docs that may have been archived since last index.
        staging_dir = Path("data/chroma") / f"family_{self.family_id}_staging"
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        staging_dir.mkdir(parents=True, exist_ok=True)

        # Map staged filename -> DB doc so we can re-tag the loaded pages
        path_to_doc = {}
        for doc_meta in docs_meta:
            src = Path(doc_meta.file_path)
            if not src.exists():
                if self.verbose:
                    print(f"[FamilyQA]   Skip missing file: {src}")
                continue
            # Stage as a unique filename so two docs with the same
            # original_filename don't collide
            staged_name = f"doc{doc_meta.id}_{src.name}"
            dst = staging_dir / staged_name
            try:
                shutil.copy2(src, dst)
                path_to_doc[str(dst.resolve())] = doc_meta
                path_to_doc[staged_name] = doc_meta  # also key by basename
            except Exception as e:
                if self.verbose:
                    print(f"[FamilyQA]   Could not stage {src.name}: {e}")

        if not path_to_doc:
            if self.verbose:
                print(f"[FamilyQA] No stage-able files. Nothing to index.")
            self.raw_docs = []
            self.chunks = []
            self.vector_store = None
            self.retriever = None
            self.qa_chain = None
            return

        # 3) Run the existing pipeline against the staging directory
        self.raw_docs = load_pdfs_hybrid(staging_dir)

        # 4) Stamp every loaded page with family_id + doc metadata so
        #    chat citations can map back to the right document
        for page in self.raw_docs:
            source = page.metadata.get("source", "")
            source_basename = Path(source).name
            doc_meta = (
                path_to_doc.get(source)
                or path_to_doc.get(source_basename)
                or path_to_doc.get(str(Path(source).resolve()))
            )
            page.metadata["family_id"] = self.family_id
            if doc_meta is not None:
                page.metadata["document_id"] = doc_meta.id
                page.metadata["category"] = doc_meta.category or "uncategorized"
                page.metadata["doc_type"] = doc_meta.doc_type
                # Replace the staging path with the user-friendly filename
                page.metadata["source"] = doc_meta.original_filename

        if not self.raw_docs:
            if self.verbose:
                print(f"[FamilyQA] No pages loaded after PDF processing.")
            self.chunks = []
            self.vector_store = None
            self.retriever = None
            self.qa_chain = None
            return

        # 5) Chunk + embed + index
        self.chunks = split_documents(
            self.raw_docs, self.chunk_size, self.chunk_overlap
        )

        if force_rebuild:
            _release_chroma_handles()

        self.vector_store = build_or_load_vectorstore(
            chunks=self.chunks,
            persist_dir=self.db_dir,
            embeddings=self.embeddings,
            force_rebuild=force_rebuild,
        )
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.DEFAULT_RETRIEVAL_K}
        )
        self.qa_chain = build_qa_chain(self.llm, self.retriever)

        # 6) Mark documents as indexed in the DB
        from db import get_session
        from db.models import Document
        from sqlmodel import select

        with get_session() as s:
            stmt = select(Document).where(Document.family_id == self.family_id)
            for d in s.exec(stmt).all():
                d.indexed_in_vectorstore = True
                s.add(d)
            s.flush()

        # 7) Clean up the staging directory — we don't need it after indexing
        shutil.rmtree(staging_dir, ignore_errors=True)

        if self.verbose:
            print(
                f"[FamilyQA] Indexed {len(docs_meta)} documents, "
                f"{len(self.raw_docs)} pages, {len(self.chunks)} chunks"
            )

def reindex_family(family_id: int, verbose: bool = False) -> FamilyQASystem:
    """Convenience function called by the upload UI after a successful upload.

    Rebuilds the family's Chroma collection from scratch with all current
    documents in the DB.
    """
    system = FamilyQASystem(family_id=family_id, verbose=verbose)
    system.index_documents(force_rebuild=True)
    return system
