# ui/components/chat_interface.py

import streamlit as st
from datetime import datetime
import time
from ui.components.sidebar import save_chat_history


def render_chat_interface(user, qa_system):
    """Render the main chat interface"""
    
    # Create new chat if none exists
    if st.session_state.current_chat_id is None:
        st.session_state.current_chat_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show metadata for assistant messages
                if message["role"] == "assistant" and "metadata" in message:
                    with st.expander("📊 Response Details"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Latency", f"{message['metadata'].get('latency_ms', 0):.0f} ms")
                        with col2:
                            st.metric("Sources", message['metadata'].get('source_count', 0))
                        with col3:
                            grounded = message['metadata'].get('grounded', False)
                            st.metric("Grounded", "✅" if grounded else "❌")
                        
                        if message['metadata'].get('sources'):
                            st.markdown("**Source Documents:**")
                            for src in message['metadata']['sources'][:5]:
                                st.markdown(f"- {src.get('source', 'Unknown')} (page {src.get('page', '?')})")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your financial documents..."):
        # Add user message to chat
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            with st.spinner("🤔 Thinking..."):
                t0 = time.perf_counter()
                
                try:
                    answer = qa_system.ask(
                        prompt,
                        disclosure_mode=st.session_state.disclosure_mode,
                        authorized=True,  # User is logged in
                        include_sources=True
                    )
                    
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    
                    # Get metadata
                    trace = getattr(qa_system, "last_trace", {})
                    
                    # Display answer
                    message_placeholder.markdown(answer)
                    
                    # Add assistant message to chat
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "timestamp": datetime.now().isoformat(),
                        "metadata": {
                            "latency_ms": latency_ms,
                            "grounded": trace.get("grounded", False),
                            "sources": trace.get("sources", []),
                            "source_count": len(trace.get("sources", []))
                        }
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    message_placeholder.markdown(error_msg)
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg,
                        "timestamp": datetime.now().isoformat()
                    })
        
        # Save chat history
        save_chat_history(user, st.session_state.current_chat_id, st.session_state.chat_history)
        
        # Rerun to update UI
        st.rerun()