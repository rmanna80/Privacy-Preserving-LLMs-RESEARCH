# ui/components/document_manager.py
"""
Document manager:
  SUPER_ADMIN → sees all advisors and all clients, can reassign clients
  ADVISOR     → sees only their own clients, uploads docs for them
  CLIENT      → read-only view of their documents
"""

import streamlit as st
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from ui.auth import AuthSystem, UserRole, User


def render_document_manager(user: User):
    if user.is_admin():
        _render_admin_view(user)
    elif user.is_advisor():
        _render_advisor_view(user)
    else:
        _render_client_view(user)


# ── Super Admin ────────────────────────────────────────────────────────────

def _render_admin_view(user: User):
    st.markdown("## 🔑 Admin — System Overview")
    st.markdown("---")

    auth     = AuthSystem()
    advisors = auth.get_all_advisors()
    clients  = auth.get_all_clients()

    col1, col2 = st.columns(2)
    col1.metric("Total Advisors", len(advisors))
    col2.metric("Total Clients",  len(clients))

    st.markdown("### 👥 Advisors")
    for adv in advisors:
        adv_clients = auth.get_clients_for_advisor(adv.username)
        with st.expander(f"📋 {adv.client_name} ({adv.username}) — {len(adv_clients)} client(s)"):
            for c in adv_clients:
                st.markdown(f"  • {c.client_name} ({c.username})")

    st.markdown("---")
    st.markdown("### ➕ Create New User")
    _render_create_user_form(auth, advisors)

    st.markdown("---")
    st.markdown("### 🔀 Reassign Client")
    _render_reassign_form(auth, clients, advisors)


def _render_create_user_form(auth: AuthSystem, advisors):
    with st.form("create_user_form"):
        col1, col2 = st.columns(2)
        with col1:
            new_email    = st.text_input("Email")
            new_name     = st.text_input("Full Name")
        with col2:
            new_role     = st.selectbox("Role", ["client", "advisor"])
            new_password = st.text_input("Temporary Password", type="password")

        advisor_options = {a.client_name: a.username for a in advisors}
        selected_advisor = None
        if new_role == "client" and advisor_options:
            selected_advisor = st.selectbox("Assign to Advisor", list(advisor_options.keys()))

        if st.form_submit_button("✅ Create User", type="primary"):
            if not all([new_email, new_name, new_password]):
                st.error("All fields are required.")
            else:
                advisor_id = advisor_options.get(selected_advisor) if selected_advisor else None
                ok = auth.create_user(
                    username=new_email.strip().lower(),
                    password=new_password,
                    role=new_role,
                    name=new_name,
                    advisor_id=advisor_id,
                )
                if ok:
                    st.success(f"✅ Created {new_role} account for {new_name}")
                    st.rerun()
                else:
                    st.error("User already exists or creation failed.")


def _render_reassign_form(auth: AuthSystem, clients, advisors):
    if not clients or not advisors:
        st.info("Need at least one client and one advisor to reassign.")
        return

    with st.form("reassign_form"):
        client_options  = {f"{c.client_name} ({c.username})": c.username for c in clients}
        advisor_options = {a.client_name: a.username for a in advisors}

        selected_client  = st.selectbox("Client",      list(client_options.keys()))
        selected_advisor = st.selectbox("New Advisor", list(advisor_options.keys()))

        if st.form_submit_button("🔀 Reassign"):
            ok = auth.reassign_client(
                client_options[selected_client],
                advisor_options[selected_advisor],
            )
            if ok:
                st.success("Client reassigned successfully.")
                st.rerun()
            else:
                st.error("Reassignment failed.")


# ── Advisor ────────────────────────────────────────────────────────────────

def _render_advisor_view(user: User):
    st.markdown("## 📁 Client Document Manager")
    st.markdown("Upload and manage documents for your clients.")
    st.markdown("---")

    auth    = AuthSystem()
    clients = auth.get_clients_for_advisor(user.username)

    if not clients:
        st.info("You have no clients assigned yet. Contact your administrator.")
        return

    client_options   = {c.client_name: c for c in clients}
    selected_name    = st.selectbox("📋 Select Client", list(client_options.keys()))
    selected_client  = client_options[selected_name]
    client_dir       = auth.get_client_documents_dir(selected_client.username)

    st.markdown(f"### Documents for **{selected_client.client_name}**")

    # Existing docs
    existing_pdfs = sorted(client_dir.glob("*.pdf"))
    if existing_pdfs:
        st.markdown(f"**{len(existing_pdfs)} document(s) on file:**")
        for pdf in existing_pdfs:
            c1, c2 = st.columns([5, 1])
            c1.markdown(f"📄 {pdf.name}")
            if c2.button("🗑️", key=f"del_{pdf.name}_{selected_client.username}",
                         help=f"Delete {pdf.name}"):
                pdf.unlink()
                st.session_state.qa_system = None   # force reindex
                st.success(f"Deleted {pdf.name}")
                st.rerun()
    else:
        st.info(f"No documents uploaded for {selected_client.client_name} yet.")

    st.markdown("---")
    st.markdown("### ⬆️ Upload Documents")

    uploaded_files = st.file_uploader(
        f"Upload PDFs for {selected_client.client_name}",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"upload_{selected_client.username}",
    )

    if uploaded_files:
        if st.button("💾 Save", type="primary"):
            saved = []
            for f in uploaded_files:
                (client_dir / f.name).write_bytes(f.getvalue())
                saved.append(f.name)
            st.success(f"✅ Saved {len(saved)} file(s) for {selected_client.client_name}")
            st.session_state.qa_system = None   # force reindex on next client login
            st.rerun()


# ── Client ─────────────────────────────────────────────────────────────────

def _render_client_view(user: User):
    st.markdown("## 📁 My Documents")
    st.markdown("These documents have been shared with you by your financial advisor.")
    st.markdown("---")

    auth       = AuthSystem()
    client_dir = auth.get_client_documents_dir(user.username)
    pdfs       = sorted(client_dir.glob("*.pdf"))

    if pdfs:
        st.markdown(f"**{len(pdfs)} document(s) on file:**")
        for pdf in pdfs:
            st.markdown(f"📄 {pdf.name}")
    else:
        st.info("📭 Your advisor hasn't uploaded any documents for you yet.")
        st.markdown("Please contact your financial advisor to get started.")