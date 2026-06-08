"""
ui/chat_bridge.py

A tiny bridge module that reuses the existing initialize_qa_system() from
ui/app.py — but exposes it without creating a circular import (since app.py
imports components, and the new chat-enabled components need to call back
into app.py's logic).

Why this file exists
--------------------
The existing initialize_qa_system() in ui/app.py works perfectly. But:
  - app.py imports advisor_shell, client_portal, etc.
  - If those new chat-using components tried to import from app.py,
    we'd have a circular import.

So we duplicate the small initialization logic here, in a module nobody
else imports from. Net code addition: ~40 lines, zero changes to app.py.

This also gives us one clean place to add family-scoped chat later (Phase 4).
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
) -> Optional[FinancialQASystem]:
    """Return a ready FinancialQASystem for the given context.

    Reuses an existing cached instance when the (user, client) scope hasn't
    changed. Otherwise rebuilds. Mirrors initialize_qa_system() in app.py —
    kept in sync intentionally so behavior matches the original chat tab.
    """
    auth = AuthSystem()

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

    with st.spinner("Loading AI system..."):
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
) -> str:
    """A stable key identifying the current chat scope. Used to load/save
    chat_history into chat_histories[key]."""
    if user.is_advisor() and selected_client_username:
        return f"advisor::{user.username}::client::{selected_client_username}"
    return f"user_{user.username}"


def load_chat_for_context(
    user,
    selected_client_username: Optional[str] = None,
) -> None:
    """Pull the right history into st.session_state.chat_history."""
    key = chat_context_key(user, selected_client_username)
    st.session_state.chat_history = (
        st.session_state.chat_histories.get(key, []).copy()
    )
    st.session_state.current_chat_id = key


def save_chat_for_context(
    user,
    selected_client_username: Optional[str] = None,
) -> None:
    """Persist current chat_history into chat_histories[key]."""
    key = chat_context_key(user, selected_client_username)
    st.session_state.chat_histories[key] = (
        st.session_state.get("chat_history", []).copy()
    )
    st.session_state.current_chat_id = key
