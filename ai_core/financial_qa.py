# ai_core/financial_qa.py

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document

from .pdf_loader import load_pdfs, split_documents
from .embeddings import LocalEmbeddings
from .vector_store import build_or_load_vectorstore
from .llm_ollama import build_ollama_llm
from .qa_chain import build_qa_chain
from .privacy_policy import PrivacyPolicy, DisclosureMode


class FinancialQASystem:
    """
    Loads PDFs, chunks them, builds/loads a Chroma vector store, and supports:
    - search(): retrieval-only debugging/evaluation
    - ask(): retrieval + local LLM answer with privacy policy enforcement
    """

    def __init__(
        self,
        docs_dir: str | Path,
        db_dir: str | Path = "vectorstore/chroma_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        verbose: bool = True,
    ) -> None:
        self.docs_dir = Path(docs_dir)
        self.db_dir = Path(db_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.verbose = verbose

        self.raw_docs: List[Document] = []
        self.chunks: List[Document] = []
        self.embeddings: Optional[LocalEmbeddings] = None
        self.vector_store = None
        self.retriever = None
        self.llm = None
        self.qa_chain = None

        self.last_trace: Dict[str, Any] = {}

    def index_documents(self, force_rebuild: bool = False) -> None:
        """Load PDFs, split into chunks, build/load the vector store, and initialize QA."""
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
            print("Initializing embeddings...")
        self.embeddings = LocalEmbeddings()

        if self.verbose:
            print("Building or loading Chroma vector store...")
        self.vector_store = build_or_load_vectorstore(
            chunks=self.chunks,
            persist_dir=self.db_dir,
            embeddings=self.embeddings,
            force_rebuild=force_rebuild,
        )

        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        self.llm = build_ollama_llm()
        self.qa_chain = build_qa_chain(self.llm, self.retriever)

        if self.verbose:
            print("Indexing complete.")
            print(f"Raw docs loaded: {len(self.raw_docs)}")
            print(f"Chunks created: {len(self.chunks)}")

    def preview_chunk(self, idx: int = 0, length: int = 400) -> None:
        """Print a preview of one document chunk for debugging."""
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

    def search(self, query: str, k: int = 4):
        """Retrieve the top-k most relevant chunks for a query (no LLM)."""
        if self.vector_store is None:
            raise RuntimeError("Vector store not built. Call index_documents() first.")

        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query)

        for i, d in enumerate(docs, start=1):
            src = d.metadata.get("source")
            page = d.metadata.get("page")
            print(f"\nResult {i}: source={src}, page={page}")
            print(d.page_content[:500])

        return docs

    def ask(
        self,
        question: str,
        disclosure_mode: DisclosureMode = DisclosureMode.AUTHORIZED,
        include_sources: bool = True,
    ) -> str:
        """Answer a question using retrieval + local LLM, then enforce privacy policy."""
        if self.qa_chain is None:
            raise RuntimeError("QA chain not ready. Call index_documents() first.")

        result = self.qa_chain({"query": question})
        answer = result.get("result", "")
        source_docs = result.get("source_documents", [])

        grounded = len(source_docs) > 0

        answer = PrivacyPolicy.enforce(
            text=answer,
            mode=disclosure_mode,
            grounded=grounded,
        )

        # Create sources for trace + optional printing
        sources = []
        for d in source_docs:
            sources.append(
                {
                    "source": d.metadata.get("source"),
                    "page": d.metadata.get("page"),
                }
            )

        # Store trace for logging/experiments
        self.last_trace = {
            "question": question,
            "disclosure_mode": disclosure_mode.value,
            "grounded": grounded,
            "sources": sources,
        }

        if include_sources and source_docs:
            seen = set()
            citations = []
            for d in source_docs:
                src = d.metadata.get("source")
                page = d.metadata.get("page")
                tag = f"{src} (page {page})"
                if tag not in seen:
                    seen.add(tag)
                    citations.append(tag)

            if citations:
                answer += "\n\nSources:\n" + "\n".join(f"- {c}" for c in citations)

        return answer
