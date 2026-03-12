# ui/components/analytics_dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
from datetime import datetime


def render_analytics_dashboard():
    """Render research analytics dashboard"""
    
    st.markdown("# 📊 Research Analytics Dashboard")
    
    tabs = st.tabs([
        "📈 Performance Metrics",
        "🔒 Privacy Compliance", 
        "📉 Accuracy Analysis",
        "💾 System Stats"
    ])
    
    with tabs[0]:
        render_performance_metrics()
    
    with tabs[1]:
        render_privacy_metrics()
    
    with tabs[2]:
        render_accuracy_analysis()
    
    with tabs[3]:
        render_system_stats()


def render_performance_metrics():
    """Performance analysis"""
    
    st.markdown("## ⚡ Query Performance")
    
    # Load audit logs
    logs = load_audit_logs()
    
    if not logs:
        st.info("No query data available yet. Start asking questions to generate metrics!")
        return
    
    df = pd.DataFrame(logs)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Queries",
            len(df),
            help="Total number of questions asked"
        )
    
    with col2:
        avg_latency = df['latency_ms'].mean() if 'latency_ms' in df else 0
        st.metric(
            "Avg Latency",
            f"{avg_latency:.0f} ms",
            help="Average response time"
        )
    
    with col3:
        p95_latency = df['latency_ms'].quantile(0.95) if 'latency_ms' in df and len(df) > 0 else 0
        st.metric(
            "P95 Latency",
            f"{p95_latency:.0f} ms",
            help="95th percentile latency"
        )
    
    with col4:
        grounded_rate = df['grounded'].mean() if 'grounded' in df else 0
        st.metric(
            "Grounded Rate",
            f"{grounded_rate:.1%}",
            help="% of answers found in documents"
        )
    
    # Latency over time
    st.markdown("### 📉 Latency Over Time")
    if 'latency_ms' in df and len(df) > 0:
        fig = px.line(df.reset_index(), x=df.index, y='latency_ms', 
                     title="Query Latency (ms)",
                     labels={'index': 'Query Number', 'latency_ms': 'Latency (ms)'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Latency distribution
    st.markdown("### 📊 Latency Distribution")
    if 'latency_ms' in df and len(df) > 0:
        fig = px.histogram(df, x='latency_ms', nbins=20,
                          title="Latency Distribution",
                          labels={'latency_ms': 'Latency (ms)'})
        st.plotly_chart(fig, use_container_width=True)


def render_privacy_metrics():
    """Privacy compliance analysis"""
    
    st.markdown("## 🔒 Privacy & Security Metrics")
    
    logs = load_audit_logs()
    
    if not logs:
        st.info("No data available yet.")
        return
    
    df = pd.DataFrame(logs)
    
    # Privacy mode usage
    st.markdown("### 🛡️ Disclosure Mode Usage")
    if 'disclosure_mode' in df:
        mode_counts = df['disclosure_mode'].value_counts()
        
        fig = px.pie(values=mode_counts.values, names=mode_counts.index,
                    title="Privacy Mode Distribution",
                    color_discrete_map={
                        'open': '#ff6b6b',
                        'authorized': '#4ecdc4',
                        'redacted': '#95e1d3'
                    })
        st.plotly_chart(fig, use_container_width=True)
    
    # PII detection rate
    st.markdown("### 🔍 PII Detection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Count SSN questions
        ssn_questions = sum(1 for log in logs if 'ssn' in log.get('question', '').lower() 
                           or 'social security' in log.get('question', '').lower())
        
        st.metric(
            "SSN Queries",
            ssn_questions,
            help="Questions asking for Social Security Numbers"
        )
    
    with col2:
        # Redaction effectiveness
        redacted_queries = sum(1 for log in logs if log.get('disclosure_mode') == 'redacted')
        st.metric(
            "Redacted Queries",
            redacted_queries,
            help="Queries with PII redaction enforced"
        )
    
    # Authorization tracking
    st.markdown("### 🔐 Authorization Tracking")
    if 'authorized' in df:
        auth_rate = df['authorized'].mean()
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = auth_rate * 100,
            title = {'text': "Authorization Rate (%)"},
            delta = {'reference': 100},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 75], 'color': "gray"},
                    {'range': [75, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        st.plotly_chart(fig, use_container_width=True)


def render_accuracy_analysis():
    """Accuracy evaluation results"""
    
    st.markdown("## 🎯 Accuracy Analysis")
    
    # Load evaluation results if available
    eval_log = Path("logs/eval_accuracy.jsonl")
    
    if not eval_log.exists():
        st.info("""
        📝 No evaluation data available yet.
        
        Run accuracy evaluation:
```bash
        python experiments/run_eval_with_accuracy.py
```
        """)
        return
    
    # Load eval data
    eval_data = []
    with open(eval_log, 'r') as f:
        for line in f:
            if line.strip():
                eval_data.append(json.loads(line))
    
    df = pd.DataFrame(eval_data)
    
    # Overall accuracy
    st.markdown("### 📊 Overall Accuracy")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'accuracy' in df:
            overall_acc = df['accuracy'].mean()
            st.metric("Overall Accuracy", f"{overall_acc:.1%}")
    
    with col2:
        if 'accuracy' in df:
            correct = (df['accuracy'] == 1.0).sum()
            st.metric("Correct Answers", f"{correct}/{len(df)}")
    
    with col3:
        if 'latency_ms' in df:
            avg_latency = df['latency_ms'].mean()
            st.metric("Avg Latency", f"{avg_latency:.0f} ms")
    
    # Accuracy by category
    st.markdown("### 📂 Accuracy by Category")
    if 'category' in df and 'accuracy' in df:
        category_acc = df.groupby('category')['accuracy'].agg(['mean', 'count'])
        category_acc.columns = ['Accuracy', 'Count']
        category_acc['Accuracy'] = category_acc['Accuracy'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(category_acc, use_container_width=True)
    
    # Detailed results
    st.markdown("### 📋 Detailed Results")
    if len(df) > 0:
        display_cols = ['question', 'expected_answer', 'actual_answer', 'accuracy', 'category']
        available_cols = [col for col in display_cols if col in df.columns]
        st.dataframe(df[available_cols], use_container_width=True)


def render_system_stats():
    """System statistics"""
    
    st.markdown("## 💾 System Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📁 Document Stats")
        
        docs_dir = Path("data/raw_pdfs")
        if docs_dir.exists():
            pdf_files = list(docs_dir.glob("*.pdf"))
            total_size = sum(f.stat().st_size for f in pdf_files) / (1024 * 1024)  # MB
            
            st.metric("Total Documents", len(pdf_files))
            st.metric("Total Size", f"{total_size:.1f} MB")
        
        # Vector store size
        vs_dir = Path("vectorstore")
        if vs_dir.exists():
            vs_size = sum(f.stat().st_size for f in vs_dir.rglob("*") if f.is_file()) / (1024 * 1024)
            st.metric("Vector Store Size", f"{vs_size:.1f} MB")
    
    with col2:
        st.markdown("### 🤖 Model Info")
        
        st.info("""
        **Current Configuration:**
        - Model: Qwen2.5:7b
        - Context: 4096 tokens
        - Temperature: 0.1
        - Chunk Size: 1200 chars
        - Chunk Overlap: 200 chars
        """)
    
    # Chat history stats
    st.markdown("### 💬 Usage Statistics")
    
    chat_history_dir = Path("data/chat_history")
    if chat_history_dir.exists():
        total_chats = sum(1 for _ in chat_history_dir.rglob("*.json"))
        st.metric("Total Chat Sessions", total_chats)


def load_audit_logs():
    """Load audit logs from JSONL"""
    log_file = Path("logs/audit.jsonl")
    
    if not log_file.exists():
        return []
    
    logs = []
    with open(log_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    logs.append(json.loads(line))
                except:
                    continue
    
    return logs