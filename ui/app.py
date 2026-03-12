# ui/app.py

import streamlit as st
from pathlib import Path
import sys
import time
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from ai_core import FinancialQASystem
from ai_core.privacy_policy import DisclosureMode
from ui.auth import AuthSystem, UserRole
from ui.components.sidebar import render_sidebar
from ui.components.chat_interface import render_chat_interface
from ui.components.document_manager import render_document_manager

# Page config
st.set_page_config(
    page_title="FinancialQA AI - Privacy-Preserving Financial Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stChatMessage { background-color: #1e1e1e; border-radius: 10px; padding: 15px; margin: 10px 0; }
    .stChatMessage[data-testid="user-message"] { background-color: #2b5278; }
    .stChatMessage[data-testid="assistant-message"] { background-color: #1e1e1e; }
    .stTextInput input { background-color: #262730; color: white; border: 1px solid #404040; border-radius: 8px; }
    .css-1d391kg { background-color: #0e1117; }
    h1, h2, h3 { color: #ffffff; }
    .privacy-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 10px 20px; border-radius: 20px; color: white;
        font-weight: bold; display: inline-block; margin: 10px 0;
    }
    .login-container {
        max-width: 400px; margin: 100px auto; padding: 40px;
        background-color: #1e1e1e; border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user" not in st.session_state:
        st.session_state.user = None
    if "qa_system" not in st.session_state:
        st.session_state.qa_system = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None
    if "disclosure_mode" not in st.session_state:
        st.session_state.disclosure_mode = DisclosureMode.AUTHORIZED


def login_page():
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown("# 🔒 FinancialQA AI")
    st.markdown("### Privacy-Preserving Financial Assistant")
    st.markdown('<div class="privacy-badge">🛡️ 100% Local Processing</div>', unsafe_allow_html=True)
    st.markdown("---")

    username = st.text_input("Email", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔓 Sign In", use_container_width=True):
            auth = AuthSystem()
            user = auth.authenticate(username, password)
            if user:
                st.session_state.authenticated = True
                st.session_state.user = user
                st.rerun()
            else:
                st.error("❌ Invalid credentials")

    with col2:
        if st.button("ℹ️ Demo Info", use_container_width=True):
            st.info("""
            **Demo Credentials:**

            **Advisor:**
            - Email: advisor@demo.com
            - Password: advisor123

            **Client (John Smith):**
            - Email: john.smith@demo.com
            - Password: client123
            """)

    st.markdown('</div>', unsafe_allow_html=True)


def initialize_qa_system(user):
    """Initialize QA system for the logged-in user"""
    if st.session_state.qa_system is None:
        with st.spinner("🔧 Initializing AI system..."):
            auth = AuthSystem()
            docs_dir = auth.get_user_documents_dir(user)

            system = FinancialQASystem(
                docs_dir=str(docs_dir),
                db_dir=str(PROJECT_ROOT / "vectorstore" / user.username / "chroma_db"),
                chunk_size=1200,
                chunk_overlap=200,
                verbose=False
            )

            system.index_documents(force_rebuild=False)
            st.session_state.qa_system = system

            if system.vector_store is None:
                if user.role == UserRole.CLIENT:
                    st.warning("📭 No documents found. Your advisor hasn't uploaded your documents yet.")
                else:
                    st.warning("⚠️ No documents found in advisor pool. Add PDFs to data/raw_pdfs/")
            else:
                st.success("✅ System ready!")


def main_app():
    user = st.session_state.user

    # Initialize QA system
    initialize_qa_system(user)

    # Sidebar
    with st.sidebar:
        render_sidebar(user)

    # Tabs — advisor gets 3, clients get 2 (Chat + read-only My Documents)
    if user.role == UserRole.ADVISOR:
        tab1, tab2, tab3 = st.tabs(["💬 Chat", "📁 Client Documents", "📊 Analytics"])
    else:
        tab1, tab2 = st.tabs(["💬 Chat", "📁 My Documents"])

    # Chat Tab
    with tab1:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("# 💬 Chat")
            st.markdown(f"**{user.client_name}** ({user.role.value})")
        with col2:
            mode_options = {
                "🔓 Open": DisclosureMode.OPEN,
                "✅ Authorized": DisclosureMode.AUTHORIZED,
                "🔒 Redacted": DisclosureMode.REDACTED
            }
            selected_mode = st.selectbox("Privacy Mode", options=list(mode_options.keys()), index=1)
            st.session_state.disclosure_mode = mode_options[selected_mode]

        st.markdown("---")
        render_chat_interface(user, st.session_state.qa_system)

    # Documents Tab
    with tab2:
        render_document_manager(user)

    # Analytics Tab (Advisor only)
    if user.role == UserRole.ADVISOR:
        with tab3:
            from ui.components.analytics_dashboard import render_analytics_dashboard
            render_analytics_dashboard()


def main():
    init_session_state()
    if not st.session_state.authenticated:
        login_page()
    else:
        main_app()


if __name__ == "__main__":
    main()