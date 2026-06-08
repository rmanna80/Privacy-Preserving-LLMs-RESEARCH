"""
AI routing policy.

Single source of truth for "which LLM should this operation use?"

Rule (from architectural decision, May 2026):
  - If the operation touches client documents, PII, or family-specific data
    that could identify a real person/entity, route to LOCAL (Ollama).
  - If the operation is generic platform work with NO client data exposure,
    cloud is permitted.

When in doubt, route LOCAL. Privacy posture is the wedge of this product;
breaking it once destroys the story.

Today, only the LOCAL path is wired (Ollama). The cloud paths are stubs
that raise NotImplementedError. When you're ready to add cloud, fill in
the _CLOUD_* providers and the policy below already knows what to do.

Usage:
    from ai_core.ai_routing import get_llm, Purpose

    llm = get_llm(Purpose.DOCUMENT_EXTRACTION)   # → local
    llm = get_llm(Purpose.GENERIC_DEFINITION)    # → cloud (when wired)
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class Purpose(str, Enum):
    """Why is this LLM call happening? Drives routing."""

    # ───── Sensitive — MUST be local ─────────────────────────────────
    DOCUMENT_EXTRACTION = "document_extraction"
    """Pulling SSNs, EINs, names, roles out of a real client document."""

    DOCUMENT_QA = "document_qa"
    """Chat answering 'what does this document say about X?'"""

    DOCUMENT_SUMMARY = "document_summary"
    """Summarizing a specific client's document."""

    FAMILY_INSIGHTS = "family_insights"
    """Advisor Copilot suggestions referencing a real family."""

    REPORT_GENERATION = "report_generation"
    """Estate report, balance sheet text — references real client data."""

    FAMILY_TREE_DATA = "family_tree_data"
    """Anything that touches actual people in the family graph."""

    # ───── Safe — cloud is fine (when wired) ─────────────────────────
    GENERIC_DEFINITION = "generic_definition"
    """'What is a GRAT?' — public knowledge."""

    UI_SUGGESTION = "ui_suggestion"
    """'Suggest a layout for the balance sheet page.' — no client data."""

    EMAIL_TEMPLATE = "email_template"
    """Generic template drafts with no client specifics."""

    PUBLIC_RESEARCH = "public_research"
    """Tax law updates, IRS exemption changes, market commentary."""


# Policy table: which Purpose routes to which provider.
# Edit here, not in calling code.
_ROUTING: dict[Purpose, str] = {
    # Local — sensitive
    Purpose.DOCUMENT_EXTRACTION: "local",
    Purpose.DOCUMENT_QA: "local",
    Purpose.DOCUMENT_SUMMARY: "local",
    Purpose.FAMILY_INSIGHTS: "local",
    Purpose.REPORT_GENERATION: "local",
    Purpose.FAMILY_TREE_DATA: "local",
    # Cloud — safe (currently fall back to local since cloud isn't wired)
    Purpose.GENERIC_DEFINITION: "cloud",
    Purpose.UI_SUGGESTION: "cloud",
    Purpose.EMAIL_TEMPLATE: "cloud",
    Purpose.PUBLIC_RESEARCH: "cloud",
}


def get_llm(purpose: Purpose) -> Any:
    """Return the right LLM client for this purpose.

    For now, cloud isn't wired — those purposes also fall back to local
    so nothing breaks. When you add cloud providers, replace the cloud
    branch with the actual client.
    """
    target = _ROUTING.get(purpose, "local")

    if target == "local":
        return _get_local_llm()

    if target == "cloud":
        # Until cloud is wired, fall back to local. NEVER raise here —
        # we want graceful degradation, not crashed pages.
        return _get_local_llm()

    # Belt-and-suspenders: unknown target → local.
    return _get_local_llm()


def is_local(purpose: Purpose) -> bool:
    """Predicate version, for code that needs to branch on routing."""
    return _ROUTING.get(purpose, "local") == "local"


# ───── Providers ─────────────────────────────────────────────────────

_local_singleton: Any = None


def _get_local_llm() -> Any:
    """Return the cached Ollama client.

    Lazy-imports so callers that never hit a local purpose don't pay
    the Ollama initialization cost.
    """
    global _local_singleton
    if _local_singleton is None:
        # Wire to your actual ai_core LLM here. For now, return a sentinel
        # that callers can recognize but that won't crash on import.
        try:
            from langchain_ollama import ChatOllama
            _local_singleton = ChatOllama(model="llama3.1:8b")
        except Exception:
            # If Ollama isn't available (CI, etc.), return a sentinel.
            _local_singleton = _UnavailableLLM("local Ollama not reachable")
    return _local_singleton


class _UnavailableLLM:
    """Stand-in that surfaces a clear error if someone tries to call it."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def invoke(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(f"LLM unavailable: {self.reason}")

    def __repr__(self) -> str:
        return f"<UnavailableLLM: {self.reason}>"
