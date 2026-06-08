from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import re

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
    - document indexing
    - conversational QA
    - privacy policy enforcement
    - sensitive-field routing
    - trace metadata for debugging/auditing

    This version keeps the original architecture, but breaks the logic into
    smaller methods so the file is easier to maintain and extend.
    """

    DEFAULT_RETRIEVAL_K = 15
    SENSITIVE_RETRIEVAL_K = 25
    SEARCH_K = 4

    def __init__(
        self,
        docs_dir: str | Path,
        db_dir: str | Path = "vectorstore/chroma_db",
        chunk_size: int = 1500,
        chunk_overlap: int = 300,
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

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def index_documents(self, force_rebuild: bool = False) -> None:
        if self.verbose:
            print(f"[INFO] Loading from: {self.docs_dir.resolve()}")

        self.raw_docs = load_pdfs_hybrid(self.docs_dir)
        self.embeddings = LocalEmbeddings()
        self.llm = build_ollama_llm()

        if not self.raw_docs:
            if self.verbose:
                print("[INFO] No documents found. System standing by for uploads.")
            self.chunks = []
            self.vector_store = None
            self.retriever = None
            self.qa_chain = None
            return

        self.chunks = split_documents(
            self.raw_docs,
            self.chunk_size,
            self.chunk_overlap,
        )

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

        if self.verbose:
            print(f"[INFO] Indexed {len(self.raw_docs)} pages → {len(self.chunks)} chunks")

    # ------------------------------------------------------------------
    # Public query entrypoint
    # ------------------------------------------------------------------
    def ask(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        disclosure_mode: DisclosureMode = DisclosureMode.AUTHORIZED,
        include_sources: bool = True,
        authorized: bool = False,
    ) -> str:
        chat_history = chat_history or []

        readiness_error = self._ensure_ready()
        if readiness_error:
            return readiness_error

        if looks_like_ssn_question(question):
            answer, grounded, source_docs = self._handle_sensitive_question(question)
        else:
            answer, grounded, source_docs = self._handle_general_question(
                question=question,
                chat_history=chat_history,
            )

        policy_decision = PrivacyPolicy.evaluate_disclosure(
            text=answer,
            mode=disclosure_mode,
            grounded=grounded,
            authorized=authorized,
        )
        final_answer = policy_decision.output_text

        self.last_trace = self._build_trace(
            question=question,
            disclosure_mode=disclosure_mode,
            authorized=authorized,
            grounded=grounded,
            source_docs=source_docs,
            policy_decision=policy_decision,
        )

        if include_sources and source_docs:
            citations = self._format_sources(source_docs)
            if citations:
                final_answer += "\n\n**Sources:**\n" + "\n".join(
                    f"- {citation}" for citation in citations
                )

        return final_answer

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def _ensure_ready(self) -> Optional[str]:
        if self.vector_store is None:
            return (
                "⚠️ No documents have been loaded yet. "
                "Please ask your advisor to upload your documents."
            )

        if self.qa_chain is None:
            return "⚠️ System not ready. Please try again in a moment."

        return None

    def _handle_sensitive_question(self, question: str):
    # """ For sensitive data like SSNs:
    # DO NOT rely only on retrieval.
    # Scan ALL chunks for maximum recall. """

        # 🔥 KEY FIX: use ALL chunks instead of retriever
        source_docs = self.chunks if self.chunks else []

        pairs = build_name_ssn_pairs_from_docs(source_docs)
        requested = extract_requested_name(question)

        if requested:
            matched = best_name_match(requested, list(pairs.keys()))
            if matched and matched in pairs:
                return pairs[matched], True, source_docs

            return (
                "I cannot find an SSN associated with that name in the documents.",
                False,
                source_docs,
            )

        if pairs:
            names = "\n- ".join(sorted(pairs.keys()))
            return (
                "I found SSNs for the following people in the documents:\n"
                f"- {names}\n\n"
                "Who would you like the SSN for?",
                True,
                source_docs,
            )

        return "I cannot find any SSNs in the documents.", False, source_docs

    def _handle_general_question(
        self,
        question: str,
        chat_history: List[Dict[str, str]],
    ) -> Tuple[str, bool, List[Document]]:
        result = self.qa_chain({"query": question, "chat_history": chat_history})
        answer = (result.get("result") or "").strip()
        source_docs = result.get("source_documents", []) or []
        grounded = self._is_answer_supported(question, answer, source_docs)
        return answer, grounded, source_docs

    def _is_answer_supported(
        self,
        question: str,
        answer: str,
        source_docs: List[Document],
    ) -> bool:
        """
        A stronger grounding check than `len(source_docs) > 0`.

        Current rule:
        - must have a non-empty answer
        - must have at least one retrieved source
        - must have some keyword overlap between answer and retrieved content

        This is still lightweight, but safer than assuming retrieval alone means support.
        """
        if not answer or not source_docs:
            return False

        answer_terms = self._keyword_set(answer)
        if not answer_terms:
            return False

        combined_source_text = " ".join(doc.page_content for doc in source_docs)
        source_terms = self._keyword_set(combined_source_text)

        overlap = answer_terms.intersection(source_terms)
        return len(overlap) >= 2

    def _keyword_set(self, text: str) -> set[str]:
        stopwords = {
            "the", "and", "for", "that", "with", "this", "from", "have",
            "your", "about", "what", "when", "where", "which", "into",
            "their", "there", "been", "were", "will", "would", "could",
            "should", "than", "then", "them", "they", "does", "did",
            "are", "was", "is", "be", "to", "of", "in", "on", "a", "an",
        }
        tokens = re.findall(r"\b[a-zA-Z0-9_%-]{3,}\b", text.lower())
        return {token for token in tokens if token not in stopwords}

    def _build_trace(
        self,
        question: str,
        disclosure_mode: DisclosureMode,
        authorized: bool,
        grounded: bool,
        source_docs: List[Document],
        policy_decision: Any,
    ) -> Dict[str, Any]:
        return {
            "question": question,
            "disclosure_mode": disclosure_mode.value,
            "authorized": authorized,
            "grounded": grounded,
            "sources": [
                {
                    "source": doc.metadata.get("source"),
                    "page": doc.metadata.get("page"),
                }
                for doc in source_docs
            ],
            "policy_allowed": getattr(policy_decision, "allowed", None),
            "policy_reasons": list(getattr(policy_decision, "reasons", [])),
        }

    def _format_sources(self, source_docs: List[Document]) -> List[str]:
        seen = set()
        citations: List[str] = []

        for doc in source_docs:
            tag = f"{doc.metadata.get('source')} (page {doc.metadata.get('page')})"
            if tag not in seen:
                seen.add(tag)
                citations.append(tag)

        return citations

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def search(self, query: str, k: int = SEARCH_K):
        if self.vector_store is None:
            raise RuntimeError("Vector store not built.")

        docs = self.vector_store.as_retriever(search_kwargs={"k": k}).invoke(query)
        for i, doc in enumerate(docs, 1):
            print(f"\nResult {i}: {doc.metadata.get('source')} p{doc.metadata.get('page')}")
            print(doc.page_content[:500])
        return docs

    def preview_chunk(self, idx: int = 0, length: int = 400) -> None:
        if not self.chunks:
            print("No chunks loaded.")
            return

        chunk = self.chunks[idx]
        print(f"--- Chunk {idx} ---\n{chunk.page_content[:length]}\n{chunk.metadata}")