# ui/components/document_manager.py

import streamlit as st
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from ui.auth import AuthSystem, UserRole, User


def render_document_manager(user: User):
    """
    Advisor: can upload documents for any client, and manage raw_pdfs.
    Client:  read-only view of their documents (uploaded by advisor).
    """

    if user.role == UserRole.ADVISOR:
        _render_advisor_document_manager(user)
    else:
        _render_client_document_view(user)


def _render_advisor_document_manager(user: User):
    """Advisor view: select a client and upload documents for them."""
    st.markdown("## 📁 Document Manager")
    st.markdown("Upload documents for your clients. Clients can only view documents you upload for them.")
    st.markdown("---")

    auth = AuthSystem()
    clients = auth.get_all_clients()

    if not clients:
        st.warning("No clients found.")
        return

    # Client selector
    client_options = {c.client_name: c for c in clients}
    selected_name = st.selectbox(
        "📋 Select Client",
        options=list(client_options.keys()),
        help="Choose which client to upload documents for"
    )
    selected_client = client_options[selected_name]
    client_dir = auth.get_client_documents_dir(selected_client.username)

    st.markdown(f"### Documents for **{selected_client.client_name}**")

    # Show existing documents
    existing_pdfs = sorted(client_dir.glob("*.pdf"))
    if existing_pdfs:
        st.markdown(f"**{len(existing_pdfs)} document(s) on file:**")
        for pdf in existing_pdfs:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"📄 {pdf.name}")
            with col2:
                if st.button("🗑️ Delete", key=f"del_{pdf.name}_{selected_client.username}"):
                    pdf.unlink()
                    # Clear QA system cache so it reindexes on next load
                    if "qa_system" in st.session_state:
                        st.session_state.qa_system = None
                    st.success(f"Deleted {pdf.name}")
                    st.rerun()
    else:
        st.info(f"No documents uploaded for {selected_client.client_name} yet.")

    st.markdown("---")

    # Upload section
    st.markdown("### ⬆️ Upload New Documents")
    uploaded_files = st.file_uploader(
        f"Upload PDFs for {selected_client.client_name}",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"upload_{selected_client.username}"
    )

    if uploaded_files:
        if st.button("💾 Save Documents", type="primary"):
            saved = []
            for uploaded_file in uploaded_files:
                save_path = client_dir / uploaded_file.name
                save_path.write_bytes(uploaded_file.getvalue())
                saved.append(uploaded_file.name)

            st.success(f"✅ Saved {len(saved)} document(s) for {selected_client.client_name}:")
            for name in saved:
                st.markdown(f"  - {name}")

            # Clear QA cache so system reindexes next time this client logs in
            if "qa_system" in st.session_state:
                st.session_state.qa_system = None

            st.rerun()

    st.markdown("---")
    st.markdown("### 📂 General Document Pool (Advisor Only)")
    st.caption("Documents here are only accessible when you are logged in as advisor.")

    raw_pdfs_dir = PROJECT_ROOT / "data" / "raw_pdfs"
    raw_pdfs_dir.mkdir(parents=True, exist_ok=True)
    advisor_pdfs = sorted(raw_pdfs_dir.glob("*.pdf"))

    if advisor_pdfs:
        st.markdown(f"**{len(advisor_pdfs)} document(s) in advisor pool:**")
        for pdf in advisor_pdfs:
            st.markdown(f"📄 {pdf.name}")
    else:
        st.info("No documents in advisor pool.")


def _render_client_document_view(user: User):
    """Client view: read-only list of their documents."""
    st.markdown("## 📁 My Documents")
    st.markdown("These are the documents your financial advisor has shared with you.")
    st.markdown("---")

    auth = AuthSystem()
    client_dir = auth.get_client_documents_dir(user.username)
    existing_pdfs = sorted(client_dir.glob("*.pdf"))

    if existing_pdfs:
        st.markdown(f"**{len(existing_pdfs)} document(s) on file:**")
        for pdf in existing_pdfs:
            st.markdown(f"📄 {pdf.name}")
    else:
        st.info("📭 Your advisor hasn't uploaded any documents for you yet. Please contact your financial advisor.")