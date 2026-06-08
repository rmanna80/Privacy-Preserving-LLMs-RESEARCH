from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from ai_core import FinancialQASystem
from ai_core.privacy_policy import DisclosureMode
from ui.auth import AuthSystem
from ui.components.sidebar import render_sidebar
from ui.components.chat_interface import render_chat_interface
from ui.components.document_manager import render_document_manager
from ui.components.family_manager import render_family_manager
from ui.components.advisor_shell import render_advisor_shell


APP_TITLE = "FinancialQA AI"
APP_SUBTITLE = "Privacy-Preserving Financial Assistant"
SHOW_DEMO_ACCOUNTS = False


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state() -> None:
    defaults = {
        "authenticated": False,
        "user": None,
        "qa_system": None,
        "qa_owner": None,
        "chat_history": [],
        "current_chat_id": None,
        "disclosure_mode": DisclosureMode.AUTHORIZED,
        "selected_client_username": None,
        "chat_histories": {},  # key -> list of messages
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_session_for_logout() -> None:
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.qa_system = None
    st.session_state.qa_owner = None
    st.session_state.chat_history = []
    st.session_state.current_chat_id = None
    st.session_state.disclosure_mode = DisclosureMode.AUTHORIZED
    st.session_state.selected_client_username = None
    st.session_state.chat_histories = {}


def invalidate_qa_system() -> None:
    st.session_state.qa_system = None
    st.session_state.qa_owner = None


def _chat_key_for_user(user, selected_client_username: Optional[str] = None) -> str:
    if user.is_advisor() and selected_client_username:
        return f"advisor::{user.username}::client::{selected_client_username}"
    return f"user::{user.username}"


def _load_chat_history_for_context(user, selected_client_username: Optional[str] = None) -> None:
    key = _chat_key_for_user(user, selected_client_username)
    st.session_state.chat_history = st.session_state.chat_histories.get(key, []).copy()
    st.session_state.current_chat_id = key


def _save_chat_history_for_context(user, selected_client_username: Optional[str] = None) -> None:
    key = _chat_key_for_user(user, selected_client_username)
    st.session_state.chat_histories[key] = st.session_state.get("chat_history", []).copy()
    st.session_state.current_chat_id = key


def login_page() -> None:
    st.title(APP_TITLE)
    st.subheader(APP_SUBTITLE)
    st.caption("100% local processing")

    st.markdown("---")

    username = st.text_input("Email", key="login_username").strip()
    password = st.text_input("Password", type="password", key="login_password")

    auth = AuthSystem()

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Sign In", use_container_width=True):
            if not username or not password:
                st.error("Please enter both email and password.")
                return

            if auth.is_locked_out(username):
                st.error("Too many failed attempts. Please wait 5 minutes.")
                return

            user = auth.authenticate(username, password)
            if user:
                st.session_state.authenticated = True
                st.session_state.user = user
                st.session_state.qa_system = None
                st.session_state.qa_owner = None
                st.session_state.chat_history = []
                st.session_state.current_chat_id = None
                st.session_state.selected_client_username = None
                st.rerun()
            else:
                st.error("Invalid credentials.")

    with col2:
        if SHOW_DEMO_ACCOUNTS and st.button("Demo Accounts", use_container_width=True):
            st.info(
                """
**Super Admin**
- admin@demo.com

**Advisors**
- advisor.adam@demo.com
- advisor.jake@demo.com

**Clients**
- john.smith@demo.com
- sally.smith@demo.com
- peter.professor@demo.com
"""
            )


def initialize_qa_system(
    user,
    selected_client_username: Optional[str] = None,
) -> Optional[FinancialQASystem]:
    auth = AuthSystem()

    if user.is_advisor():
        if not selected_client_username:
            return None

        client_user = auth.get_user(selected_client_username)
        if client_user is None:
            st.error("Selected client could not be found.")
            return None

        docs_dir = auth.get_client_documents_dir(selected_client_username)
        db_dir = auth.get_vectorstore_dir(f"{user.username}__{selected_client_username}")
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

        if system.vector_store is None:
            if user.is_client():
                st.warning("Your advisor has not uploaded your documents yet.")
            elif user.is_advisor():
                st.info("No documents loaded yet for this client.")
            else:
                st.info("No documents loaded yet.")
        else:
            st.success("System ready.")

        return system


def render_topbar(user) -> None:
    col1, col2 = st.columns([6, 1])

    with col1:
        st.markdown(
            f"### {APP_TITLE} — {user.client_name} "
            f"· *{user.role.value.replace('_', ' ').title()}*"
        )

    with col2:
        if st.button("Logout", use_container_width=True):
            reset_session_for_logout()
            st.rerun()


def main_app() -> None:
    user = st.session_state.user

    # Advisors get the Angel sidebar-nav workspace.
    if user.is_advisor():
        render_advisor_shell(user)
        return

    # Clients get the Angel client portal — same architecture, simpler nav.
    if user.is_client():
        from ui.components.client_portal import render_client_portal
        render_client_portal(user)
        return

    # Admin only — original tab layout.
    with st.sidebar:
        render_sidebar(user)
        st.markdown("---")
        if st.button("Refresh AI Index", use_container_width=True):
            invalidate_qa_system()
            st.rerun()

    render_topbar(user)

    tab1, tab2 = st.tabs(["User Management", "Analytics"])
    with tab1:
        render_document_manager(user)
    with tab2:
        from ui.components.analytics_dashboard import render_analytics_dashboard
        render_analytics_dashboard()

def render_advisor_chat_tab(user) -> None:
    auth = AuthSystem()
    clients = auth.get_clients_for_advisor(user.username)

    st.markdown("# Chat")
    st.markdown(f"**{user.client_name}** · *{user.role.value.replace('_', ' ').title()}*")

    if not clients:
        st.info("You have no clients assigned yet.")
        return

    client_options = {f"{c.client_name} ({c.username})": c.username for c in clients}

    existing_selected = st.session_state.get("selected_client_username")
    if existing_selected is None and clients:
        existing_selected = clients[0].username
        st.session_state.selected_client_username = existing_selected

    labels = list(client_options.keys())
    default_index = 0
    for idx, label in enumerate(labels):
        if client_options[label] == existing_selected:
            default_index = idx
            break

    col1, col2 = st.columns([3, 1])

    with col1:
        selected_label = st.selectbox("Select Client", labels, index=default_index)
        selected_client_username = client_options[selected_label]

    with col2:
        mode_options = {
            "Open": DisclosureMode.OPEN,
            "Authorized": DisclosureMode.AUTHORIZED,
            "Redacted": DisclosureMode.REDACTED,
        }
        mode_labels = list(mode_options.keys())
        current_mode = st.session_state.get("disclosure_mode", DisclosureMode.AUTHORIZED)
        current_index = mode_labels.index(
            next(label for label, value in mode_options.items() if value == current_mode)
        )
        selected_mode_label = st.selectbox("Privacy Mode", mode_labels, index=current_index)
        st.session_state.disclosure_mode = mode_options[selected_mode_label]

    # If advisor switches clients, swap chat history and invalidate client-specific QA
    previous_client_username = st.session_state.get("selected_client_username")
    if previous_client_username != selected_client_username:
        if previous_client_username is not None:
            _save_chat_history_for_context(user, previous_client_username)
        st.session_state.selected_client_username = selected_client_username
        _load_chat_history_for_context(user, selected_client_username)
        invalidate_qa_system()

    # Make sure current client's history is loaded
    if st.session_state.get("current_chat_id") != _chat_key_for_user(user, selected_client_username):
        _load_chat_history_for_context(user, selected_client_username)

    selected_client_user = auth.get_user(selected_client_username)
    if selected_client_user:
        st.caption(f"Chatting with documents for: {selected_client_user.client_name}")

    st.markdown("---")

    qa_system = initialize_qa_system(user, selected_client_username=selected_client_username)

    if qa_system is None:
        st.info("No AI system is ready for this client yet.")
        return

    render_chat_interface(user, qa_system)
    _save_chat_history_for_context(user, selected_client_username)


def render_chat_tab(user, qa_system: Optional[FinancialQASystem]) -> None:
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("# Chat")
        st.markdown(f"**{user.client_name}** · *{user.role.value.replace('_', ' ').title()}*")

    with col2:
        if not user.is_client():
            mode_options = {
                "Open": DisclosureMode.OPEN,
                "Authorized": DisclosureMode.AUTHORIZED,
                "Redacted": DisclosureMode.REDACTED,
            }
            labels = list(mode_options.keys())
            current_mode = st.session_state.get("disclosure_mode", DisclosureMode.AUTHORIZED)
            current_index = labels.index(
                next(
                    label for label, value in mode_options.items()
                    if value == current_mode
                )
            )
            selected = st.selectbox("Privacy Mode", labels, index=current_index)
            st.session_state.disclosure_mode = mode_options[selected]
        else:
            st.session_state.disclosure_mode = DisclosureMode.AUTHORIZED
            st.caption("Privacy mode is managed automatically for client access.")

    st.markdown("---")

    if qa_system is None:
        st.info("No AI system is ready for this account yet.")
        return

    _load_chat_history_for_context(user)
    render_chat_interface(user, qa_system)
    _save_chat_history_for_context(user)


def main() -> None:
    init_session_state()

    if not st.session_state.authenticated:
        login_page()
    else:
        main_app()


if __name__ == "__main__":
    main()