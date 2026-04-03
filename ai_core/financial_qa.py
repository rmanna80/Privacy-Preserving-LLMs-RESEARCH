# ai_core/financial_qa.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document

from .pdf_loader import load_pdfs_hybrid, split_documents
from .embeddings import LocalEmbeddings
from .vector_store import build_or_load_vectorstore
from .llm_ollama import build_ollama_llm
from .qa_chain import build_qa_chain
from .privacy_policy import PrivacyPolicy, DisclosureMode
from .sensitive_extractors import (
    looks_like_ssn_question,
    build_name_ssn_pairs_from_docs,
    extract_requested_name,
    best_name_match,
)


class FinancialQASystem:
    """
    RAG-based financial document QA with:
    - Conversation memory (fluid multi-turn chat)
    - Privacy policy enforcement
    - Sensitive data extraction (SSNs)
    - Graceful empty-document handling
    """

    def __init__(
        self,
        docs_dir: str | Path,
        db_dir:   str | Path = "vectorstore/chroma_db",
        chunk_size:    int = 1500,
        chunk_overlap: int = 300,
        verbose: bool = True,
    ) -> None:
        self.docs_dir      = Path(docs_dir)
        self.db_dir        = Path(db_dir)
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.verbose       = verbose

        self.raw_docs:   List[Document]        = []
        self.chunks:     List[Document]        = []
        self.embeddings: Optional[LocalEmbeddings] = None
        self.vector_store  = None
        self.retriever     = None
        self.llm           = None
        self.qa_chain      = None
        self.last_trace:   Dict[str, Any]      = {}

    # ── Indexing ───────────────────────────────────────────────────────────

    def index_documents(self, force_rebuild: bool = False) -> None:
        if self.verbose:
            print(f"[INFO] Loading from: {self.docs_dir.resolve()}")

        self.raw_docs = load_pdfs_hybrid(self.docs_dir)

        # Graceful empty-doc handling — don't crash, just stand by
        if not self.raw_docs:
            print(f"[INFO] No documents found. System standing by for uploads.")
            self.embeddings = LocalEmbeddings()
            self.llm        = build_ollama_llm()
            return

        self.chunks = split_documents(self.raw_docs, self.chunk_size, self.chunk_overlap)
        self.embeddings = LocalEmbeddings()

        self.vector_store = build_or_load_vectorstore(
            chunks=self.chunks,
            persist_dir=self.db_dir,
            embeddings=self.embeddings,
            force_rebuild=force_rebuild,
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 15})
        self.llm       = build_ollama_llm()
        self.qa_chain  = build_qa_chain(self.llm, self.retriever)

        if self.verbose:
            print(f"[INFO] Indexed {len(self.raw_docs)} pages → {len(self.chunks)} chunks")

    # ── Query ──────────────────────────────────────────────────────────────

    def ask(
        self,
        question: str,
        chat_history:    List[Dict[str, str]] = None,
        disclosure_mode: DisclosureMode = DisclosureMode.AUTHORIZED,
        include_sources: bool = True,
        authorized:      bool = False,
    ) -> str:
        """
        Answer a question with full conversation context.

        chat_history: list of {"role": "user"|"assistant", "content": str}
                      Pass the full session history for fluid multi-turn chat.
        """
        chat_history = chat_history or []

        if self.vector_store is None:
            return (
                "📭 No documents have been loaded yet. "
                "Please ask your advisor to upload your documents."
            )

        answer      = ""
        grounded    = False
        source_docs = []

        # ── SSN path ───────────────────────────────────────────────────────
        if looks_like_ssn_question(question):
            sensitive_retriever = self.vector_store.as_retriever(search_kwargs={"k": 25})
            source_docs = sensitive_retriever.invoke(question) or []
            pairs       = build_name_ssn_pairs_from_docs(source_docs)
            requested   = extract_requested_name(question)

            if requested:
                matched = best_name_match(requested, list(pairs.keys()))
                if matched and matched in pairs:
                    answer   = pairs[matched]
                    grounded = True
                else:
                    answer   = "I cannot find an SSN associated with that name in the documents."
                    grounded = False
            else:
                if pairs:
                    answer = (
                        "I found SSNs for the following people in the documents:\n- "
                        + "\n- ".join(sorted(pairs.keys()))
                        + "\n\nWho would you like the SSN for?"
                    )
                    grounded = True
                else:
                    answer   = "I cannot find any SSNs in the retrieved documents."
                    grounded = False

        # ── Normal conversational QA path ──────────────────────────────────
        else:
            if self.qa_chain is None:
                return "⚠️ System not ready. Please try again in a moment."

            result      = self.qa_chain({"query": question, "chat_history": chat_history})
            answer      = result.get("result", "")
            source_docs = result.get("source_documents", []) or []
            grounded    = len(source_docs) > 0

        # ── Privacy gate ───────────────────────────────────────────────────
        answer = PrivacyPolicy.enforce(
            text=answer,
            mode=disclosure_mode,
            grounded=grounded,
            authorized=authorized,
        )

        # ── Trace ──────────────────────────────────────────────────────────
        sources = [
            {"source": d.metadata.get("source"), "page": d.metadata.get("page")}
            for d in source_docs
        ]
        self.last_trace = {
            "question":        question,
            "disclosure_mode": disclosure_mode.value,
            "authorized":      authorized,
            "grounded":        grounded,
            "sources":         sources,
        }

        # ── Citations ──────────────────────────────────────────────────────
        if include_sources and source_docs:
            seen, citations = set(), []
            for d in source_docs:
                tag = f"{d.metadata.get('source')} (page {d.metadata.get('page')})"
                if tag not in seen:
                    seen.add(tag)
                    citations.append(tag)
            if citations:
                answer += "\n\n**Sources:**\n" + "\n".join(f"- {c}" for c in citations)

        return answer

    # ── Utilities ──────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 4):
        if self.vector_store is None:
            raise RuntimeError("Vector store not built.")
        docs = self.vector_store.as_retriever(search_kwargs={"k": k}).invoke(query)
        for i, d in enumerate(docs, 1):
            print(f"\nResult {i}: {d.metadata.get('source')} p{d.metadata.get('page')}")
            print(d.page_content[:500])
        return docs

    def preview_chunk(self, idx: int = 0, length: int = 400) -> None:
        if not self.chunks:
            print("No chunks loaded.")
            return
        c = self.chunks[idx]
        print(f"--- Chunk {idx} ---\n{c.page_content[:length]}\n{c.metadata}")