# ui/app.py
import streamlit as st
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from ai_core import FinancialQASystem
from ai_core.privacy_policy import DisclosureMode
from ui.auth import AuthSystem, UserRole
from ui.components.sidebar import render_sidebar
from ui.components.chat_interface import render_chat_interface
from ui.components.document_manager import render_document_manager

st.set_page_config(
    page_title="FinancialQA AI",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stChatMessage { background-color: #1e1e1e; border-radius: 10px; padding: 15px; margin: 10px 0; }
    .stChatMessage[data-testid="user-message"] { background-color: #2b5278; }
    h1, h2, h3 { color: #ffffff; }
    .privacy-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 8px 18px; border-radius: 20px; color: white;
        font-weight: bold; display: inline-block; margin: 8px 0;
    }
    .login-container {
        max-width: 420px; margin: 80px auto; padding: 40px;
        background-color: #1e1e1e; border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────

def init_session_state():
    defaults = {
        "authenticated":    False,
        "user":             None,
        "qa_system":        None,
        "chat_history":     [],
        "current_chat_id":  None,
        "disclosure_mode":  DisclosureMode.AUTHORIZED,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Login ──────────────────────────────────────────────────────────────────

def login_page():
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown("# 🔒 FinancialQA AI")
    st.markdown("### Privacy-Preserving Financial Assistant")
    st.markdown('<div class="privacy-badge">🛡️ 100% Local Processing</div>', unsafe_allow_html=True)
    st.markdown("---")

    username = st.text_input("Email",    key="login_username")
    password = st.text_input("Password", type="password", key="login_password")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔓 Sign In", use_container_width=True):
            auth = AuthSystem()

            if auth.is_locked_out(username):
                st.error("🔒 Too many failed attempts. Please wait 5 minutes.")
            else:
                user = auth.authenticate(username, password)
                if user:
                    st.session_state.authenticated = True
                    st.session_state.user          = user
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")

    with col2:
        if st.button("ℹ️ Demo Accounts", use_container_width=True):
            st.info("""
**Super Admin:**
- admin@demo.com / admin123

**Advisors:**
- advisor.adam@demo.com / advisor123
- advisor.jake@demo.com / advisor456

**Clients (Adam's):**
- john.smith@demo.com / client123
- sally.smith@demo.com / client123

**Clients (Jake's):**
- peter.professor@demo.com / client123
            """)

    st.markdown('</div>', unsafe_allow_html=True)


# ── QA System init ─────────────────────────────────────────────────────────

def initialize_qa_system(user):
    if st.session_state.qa_system is not None:
        return

    with st.spinner("🔧 Loading AI system..."):
        auth     = AuthSystem()
        docs_dir = auth.get_user_documents_dir(user)
        db_dir   = auth.get_vectorstore_dir(user.username)

        system = FinancialQASystem(
            docs_dir=str(docs_dir),
            db_dir=str(db_dir),
            chunk_size=1200,
            chunk_overlap=200,
            verbose=False,
        )
        system.index_documents(force_rebuild=False)
        st.session_state.qa_system = system

        if system.vector_store is None:
            if user.is_client():
                st.warning("📭 Your advisor hasn't uploaded your documents yet.")
            elif user.is_advisor():
                st.info("📂 Upload documents for your clients in the Documents tab.")
        else:
            st.success("✅ System ready!")


# ── Main app ───────────────────────────────────────────────────────────────

def main_app():
    user = st.session_state.user
    initialize_qa_system(user)

    with st.sidebar:
        render_sidebar(user)

    # Tab layout by role
    if user.is_admin():
        tab1, tab2 = st.tabs(["👥 User Management", "📊 Analytics"])
        with tab1:
            render_document_manager(user)
        with tab2:
            from ui.components.analytics_dashboard import render_analytics_dashboard
            render_analytics_dashboard()

    elif user.is_advisor():
        tab1, tab2, tab3 = st.tabs(["💬 Chat", "📁 Client Documents", "📊 Analytics"])
        with tab1:
            _render_chat_tab(user)
        with tab2:
            render_document_manager(user)
        with tab3:
            from ui.components.analytics_dashboard import render_analytics_dashboard
            render_analytics_dashboard()

    else:  # client
        tab1, tab2 = st.tabs(["💬 Chat", "📁 My Documents"])
        with tab1:
            _render_chat_tab(user)
        with tab2:
            render_document_manager(user)


def _render_chat_tab(user):
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("# 💬 Chat")
        st.markdown(f"**{user.client_name}** · *{user.role.value.replace('_', ' ').title()}*")
    with col2:
        mode_options = {
            "🔓 Open":       DisclosureMode.OPEN,
            "✅ Authorized": DisclosureMode.AUTHORIZED,
            "🔒 Redacted":   DisclosureMode.REDACTED,
        }
        selected = st.selectbox("Privacy Mode", list(mode_options.keys()), index=1)
        st.session_state.disclosure_mode = mode_options[selected]

    st.markdown("---")
    render_chat_interface(user, st.session_state.qa_system)


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    init_session_state()
    if not st.session_state.authenticated:
        login_page()
    else:
        main_app()


if __name__ == "__main__":
    main()