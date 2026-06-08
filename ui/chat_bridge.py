"""
ui/chat_bridge.py — updated for sub-phase 4c family-scoped chat.

Now supports two paths:
  - LEGACY (per-client docs folder): same as before. Used by clients that
    don't have a family_id in their session (admin docs, or transition
    state). Stays around for compatibility.
  - FAMILY-SCOPED: when a family_id is in session_state, the chat uses
    FamilyQASystem reading from data/chroma/family_<id>/, which is built
    from the new Document table via reindex_family().

Routing rule:
  If a family_id is known for the current user context, use FamilyQASystem.
  Otherwise fall back to the legacy FinancialQASystem.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

from ai_core import FinancialQASystem
from ai_core.privacy_policy import DisclosureMode
from ui.auth import AuthSystem


def ensure_chat_session_state() -> None:
    """Make sure all session-state keys the chat interface expects exist."""
    defaults = {
        "qa_system": None,
        "qa_owner": None,
        "chat_history": [],
        "current_chat_id": None,
        "disclosure_mode": DisclosureMode.AUTHORIZED,
        "chat_histories": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_or_build_qa(
    user,
    selected_client_username: Optional[str] = None,
    family_id: Optional[int] = None,
):
    """Return a ready QA system for the given context.

    If family_id is given, returns a FamilyQASystem scoped to that family's
    Document table rows. Otherwise falls back to the legacy per-client
    FinancialQASystem.

    Reuses cached instance when the scope key matches.
    """
    auth = AuthSystem()

    # ─── Family-scoped path (new in 4c) ───────────────────────────────
    if family_id is not None:
        owner_key = f"family::{family_id}"
        current_owner = st.session_state.get("qa_owner")
        current_system = st.session_state.get("qa_system")
        if current_system is not None and current_owner == owner_key:
            return current_system

        with st.spinner("Loading family AI index…"):
            from ai_core.family_qa import FamilyQASystem
            system = FamilyQASystem(family_id=family_id, verbose=False)
            system.index_documents(force_rebuild=False)
            st.session_state.qa_system = system
            st.session_state.qa_owner = owner_key
            return system

    # ─── Legacy per-client path ───────────────────────────────────────
    if user.is_advisor():
        if not selected_client_username:
            return None
        client_user = auth.get_user(selected_client_username)
        if client_user is None:
            st.error("Selected client could not be found.")
            return None
        docs_dir = auth.get_client_documents_dir(selected_client_username)
        db_dir = auth.get_vectorstore_dir(
            f"{user.username}__{selected_client_username}"
        )
        owner_key = f"advisor::{user.username}::client::{selected_client_username}"

    elif user.is_client():
        docs_dir = auth.get_client_documents_dir(user.username)
        db_dir = auth.get_vectorstore_dir(user.username)
        owner_key = f"client::{user.username}"

    else:
        docs_dir = auth.get_user_documents_dir(user)
        db_dir = auth.get_vectorstore_dir(user.username)
        owner_key = f"user::{user.username}"

    current_owner = st.session_state.get("qa_owner")
    current_system = st.session_state.get("qa_system")
    if current_system is not None and current_owner == owner_key:
        return current_system

    with st.spinner("Loading AI system…"):
        system = FinancialQASystem(
            docs_dir=str(docs_dir),
            db_dir=str(db_dir),
            chunk_size=1200,
            chunk_overlap=200,
            verbose=False,
        )
        system.index_documents(force_rebuild=False)
        st.session_state.qa_system = system
        st.session_state.qa_owner = owner_key
        return system


def chat_context_key(
    user,
    selected_client_username: Optional[str] = None,
    family_id: Optional[int] = None,
) -> str:
    """A stable key identifying the current chat scope.

    Windows-filename-safe (no colons or other reserved chars).
    """
    if family_id is not None:
        return f"family_{family_id}"
    if user.is_advisor() and selected_client_username:
        return f"advisor_{user.username}_client_{selected_client_username}"
    return f"user_{user.username}"


def load_chat_for_context(
    user,
    selected_client_username: Optional[str] = None,
    family_id: Optional[int] = None,
) -> None:
    """Pull the right history into st.session_state.chat_history."""
    key = chat_context_key(user, selected_client_username, family_id)
    st.session_state.chat_history = (
        st.session_state.chat_histories.get(key, []).copy()
    )
    st.session_state.current_chat_id = key


def save_chat_for_context(
    user,
    selected_client_username: Optional[str] = None,
    family_id: Optional[int] = None,
) -> None:
    """Persist current chat_history into chat_histories[key]."""
    key = chat_context_key(user, selected_client_username, family_id)
    st.session_state.chat_histories[key] = (
        st.session_state.get("chat_history", []).copy()
    )
    st.session_state.current_chat_id = key
