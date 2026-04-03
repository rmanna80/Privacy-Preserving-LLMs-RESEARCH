# ui/components/chat_interface.py
"""
Conversational chat interface.
Passes the full session history to the QA system so the LLM
can maintain context — just like ChatGPT / Claude.
"""

import streamlit as st
from datetime import datetime
import time
from ui.components.sidebar import save_chat_history


def render_chat_interface(user, qa_system):
    """Render the main conversational chat interface."""

    # Create a chat session ID if none exists
    if st.session_state.current_chat_id is None:
        st.session_state.current_chat_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # ── Display existing messages ─────────────────────────────────────────
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and "metadata" in message:
                meta = message["metadata"]
                with st.expander("📊 Response Details", expanded=False):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Latency",  f"{meta.get('latency_ms', 0):.0f} ms")
                    c2.metric("Sources",  meta.get("source_count", 0))
                    c3.metric("Grounded", "✅" if meta.get("grounded") else "❌")
                    if meta.get("sources"):
                        st.markdown("**Source Documents:**")
                        for s in meta["sources"][:5]:
                            st.caption(f"• {s.get('source','?')} — page {s.get('page','?')}")

    # ── Suggested prompts when chat is empty ──────────────────────────────
    if not st.session_state.chat_history:
        st.markdown("#### 💡 Try asking:")
        suggestions = [
            "What is my adjusted gross income?",
            "What filing status did we use?",
            "Summarize my tax return",
            "What documents do you have for me?",
        ]
        cols = st.columns(len(suggestions))
        for col, suggestion in zip(cols, suggestions):
            if col.button(suggestion, use_container_width=True):
                _handle_message(suggestion, user, qa_system)
                return

    # ── Chat input ────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask anything about your financial documents..."):
        _handle_message(prompt, user, qa_system)


def _handle_message(prompt: str, user, qa_system):
    """Process a user message and get an AI response."""

    # Add to history and display
    st.session_state.chat_history.append({
        "role":      "user",
        "content":   prompt,
        "timestamp": datetime.now().isoformat(),
    })
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build chat history for LLM (exclude current message — it's already in the prompt)
    history_for_llm = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.chat_history[:-1]   # all but the last (current) user message
        if m["role"] in ("user", "assistant")
    ]

    # Get AI response
    with st.chat_message("assistant"):
        placeholder = st.empty()

        with st.spinner(""):
            t0 = time.perf_counter()
            try:
                if qa_system is None or qa_system.vector_store is None:
                    answer = "📭 No documents are loaded yet. Your advisor needs to upload documents for you first."
                    latency_ms = 0
                    trace = {}
                else:
                    answer = qa_system.ask(
                        question=prompt,
                        chat_history=history_for_llm,
                        disclosure_mode=st.session_state.disclosure_mode,
                        authorized=True,   # user is authenticated
                        include_sources=False,  # we show sources in expander separately
                    )
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    trace = getattr(qa_system, "last_trace", {})

                    # Append source citations as a clean separate block
                    sources = trace.get("sources", [])
                    if sources:
                        seen, cited = set(), []
                        for s in sources:
                            tag = f"{s.get('source','?')} — page {s.get('page','?')}"
                            if tag not in seen:
                                seen.add(tag)
                                cited.append(tag)

            except Exception as e:
                answer     = f"❌ Something went wrong: {str(e)}"
                latency_ms = (time.perf_counter() - t0) * 1000.0
                trace      = {}

        placeholder.markdown(answer)

    # Store assistant message with metadata
    st.session_state.chat_history.append({
        "role":      "assistant",
        "content":   answer,
        "timestamp": datetime.now().isoformat(),
        "metadata": {
            "latency_ms":   latency_ms,
            "grounded":     trace.get("grounded", False),
            "sources":      trace.get("sources", []),
            "source_count": len(trace.get("sources", [])),
        },
    })

    # Persist chat
    save_chat_history(user, st.session_state.current_chat_id, st.session_state.chat_history)

    st.rerun()