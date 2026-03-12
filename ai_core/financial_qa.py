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
    extract_ssns,
    extract_requested_name,
    best_name_match,
)


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

    def index_documents(self, force_rebuild: bool = False) -> None:
        """Load PDFs, split into chunks, build/load the vector store, and initialize QA."""
        if self.verbose:
            print("Loading PDFs...")

         # ADD THESE DEBUG LINES
        print(f"[DEBUG] docs_dir = {self.docs_dir}")
        print(f"[DEBUG] docs_dir absolute = {self.docs_dir.resolve()}")
        print(f"[DEBUG] docs_dir exists = {self.docs_dir.exists()}")
        if self.docs_dir.exists():
            pdfs = list(self.docs_dir.glob("*.pdf"))
            print(f"[DEBUG] PDFs found = {len(pdfs)}")
            for p in pdfs[:3]:
                print(f"[DEBUG]   {p}")
        
       
        
        self.raw_docs = load_pdfs_hybrid(self.docs_dir)

         # Handle empty document directory gracefully
        if not self.raw_docs:
            print(f"[INFO] No documents found in {self.docs_dir}. Waiting for uploads.")
            self.chunks = []
            self.embeddings = LocalEmbeddings()
            self.vector_store = None
            self.retriever = None
            self.llm = build_ollama_llm()
            self.qa_chain = None
            return  # exit early, no crash 


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

        # Retriever with increased k for better recall
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 15}  # Retrieve more documents for reranking
        )

        # Build LLM with Qwen 14B
        self.llm = build_ollama_llm()
        
        # Build QA chain with reranking
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

    def search(self, query: str, k: int = 4): # k was at 4, but changed to 8
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
    authorized: bool = False,
    ) -> str:
        """Answer a question using retrieval + local LLM, then enforce privacy policy."""
        if self.vector_store is None:
            raise RuntimeError("System not ready. Call index_documents() first.")

        # Always define these so you don't get "variable referenced before assignment"
        answer = ""
        grounded = False
        source_docs = []

        if looks_like_ssn_question(question):
            # pull more context for sensitive queries
            sensitive_retriever = self.vector_store.as_retriever(search_kwargs={"k": 25})
            source_docs = sensitive_retriever.invoke(question) or []

            
            pairs = build_name_ssn_pairs_from_docs(source_docs)   # <-- docs in
            requested = extract_requested_name(question)          # <-- question in 

            # DEBUG (optional)
            print("[DEBUG] pairs:", pairs)
            print("[DEBUG] requested:", requested)
            print("[DEBUG] authorized:", authorized, "mode:", disclosure_mode.value)

            # If user didn’t specify a name, but they typed “Sally ssn” (no last name),
            # we still try a match against candidates.
            if requested:
                matched = best_name_match(requested, list(pairs.keys()))
                if matched and matched in pairs:
                    answer = pairs[matched]
                    grounded = True
                else:
                    answer = "I cannot find an SSN associated with that name in the retrieved context."
                    grounded = False
            else:
                if pairs:
                    names = sorted(pairs.keys())
                    answer = (
                        "I found SSNs for the following names in the retrieved context:\n- "
                        + "\n- ".join(names)
                        + "\n\nPlease specify which person you want."
                    )
                    grounded = True
                else:
                    answer = "I cannot find an SSN in the retrieved context."
                    grounded = False

        else:
            if self.qa_chain is None:
                raise RuntimeError("QA chain not ready. Call index_documents() first.")

            result = self.qa_chain({"query": question})
            answer = result.get("result", "")
            source_docs = result.get("source_documents", []) or []
            grounded = len(source_docs) > 0

        # privacy gate
        answer = PrivacyPolicy.enforce(
            text=answer,
            mode=disclosure_mode,
            grounded=grounded,
            authorized=authorized,
        )

        # trace
        sources = [{"source": d.metadata.get("source"), "page": d.metadata.get("page")} for d in source_docs]
        self.last_trace = {
            "question": question,
            "disclosure_mode": disclosure_mode.value,
            "authorized": authorized,
            "grounded": grounded,
            "sources": sources,
        }

        # optional citations
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
