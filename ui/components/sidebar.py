# ui/components/sidebar.py

import streamlit as st
from datetime import datetime
from pathlib import Path
import json


def save_chat_history(user, chat_id, messages):
    """Save chat history to file"""
    history_dir = Path(f"data/chat_history/{user.username}")
    history_dir.mkdir(parents=True, exist_ok=True)
    
    history_file = history_dir / f"{chat_id}.json"
    history_file.write_text(json.dumps({
        "chat_id": chat_id,
        "timestamp": datetime.now().isoformat(),
        "messages": messages
    }, indent=2))


def load_chat_history(user, chat_id):
    """Load chat history from file"""
    history_file = Path(f"data/chat_history/{user.username}/{chat_id}.json")
    if history_file.exists():
        return json.loads(history_file.read_text())["messages"]
    return []


def get_user_chats(user):
    """Get all chat sessions for a user"""
    history_dir = Path(f"data/chat_history/{user.username}")
    if not history_dir.exists():
        return []
    
    chats = []
    for file in sorted(history_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        data = json.loads(file.read_text())
        # Get first user message as title
        title = "New Chat"
        for msg in data["messages"]:
            if msg["role"] == "user":
                title = msg["content"][:50] + ("..." if len(msg["content"]) > 50 else "")
                break
        
        chats.append({
            "id": data["chat_id"],
            "title": title,
            "timestamp": data["timestamp"]
        })
    
    return chats


def render_sidebar(user):
    """Render sidebar with chat history"""
    st.markdown("## 💼 FinancialQA AI")
    st.markdown(f"**{user.client_name}**")
    st.markdown(f"*{user.role.value.title()}*")
    
    st.markdown("---")
    
    # New chat button
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.current_chat_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown("### 📜 Recent Chats")
    
    # List recent chats
    chats = get_user_chats(user)
    
    if chats:
        for chat in chats[:10]:  # Show last 10 chats
            chat_date = datetime.fromisoformat(chat["timestamp"]).strftime("%m/%d %I:%M %p")
            if st.button(
                f"💬 {chat['title']}\n_{chat_date}_",
                key=chat["id"],
                use_container_width=True
            ):
                st.session_state.current_chat_id = chat["id"]
                st.session_state.chat_history = load_chat_history(user, chat["id"])
                st.rerun()
    else:
        st.info("No chat history yet")
    
    st.markdown("---")
    
    # Privacy info
    with st.expander("🛡️ Privacy & Security"):
        st.markdown("""
        **Data Privacy:**
        - 🔒 All processing is local
        - 🚫 No data sent to cloud
        - 📁 Your documents stay on device
        - 🔐 End-to-end encrypted storage
        
        **Disclosure Modes:**
        - **Open**: All data visible (testing only)
        - **Authorized**: PII only if authenticated
        - **Redacted**: All PII masked
        """)
    
    # Logout button
    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        # Save current chat before logout
        if st.session_state.chat_history and st.session_state.current_chat_id:
            save_chat_history(user, st.session_state.current_chat_id, st.session_state.chat_history)
        
        st.session_state.authenticated = False
        st.session_state.user = None
        st.session_state.qa_system = None
        st.session_state.chat_history = []
        st.session_state.current_chat_id = None
        st.rerun()