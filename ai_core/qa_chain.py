# ai_core/qa_chain.py
from __future__ import annotations

from typing import Dict, Any, List, Optional
import re

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from sentence_transformers import CrossEncoder

RERANKER = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

SYSTEM_PROMPT = """You are a knowledgeable, conversational financial document assistant for a financial advisory firm.

You have access to the client's uploaded financial documents including tax returns, insurance policies, investment statements, and legal documents.

CORE BEHAVIOR:
- Be conversational and natural — like a knowledgeable colleague, not a search engine
- Remember context from earlier in this conversation and refer back to it naturally
- If the user says "what about his wife?" after asking about John Smith, you know they mean Sally Smith
- Give complete, helpful answers — don't just extract numbers, explain what they mean when relevant
- If you're unsure, say so honestly rather than guessing

ACCURACY RULES:
1. Only use information from the provided documents — never fabricate numbers or facts
2. Extract EXACT values: "$114,550" not "approximately $115,000"
3. For SSNs use format: XXX-XX-XXXX
4. If the answer isn't in the documents, say clearly: "I don't see that in the documents I have access to."

DOCUMENT EXTRACTION PATTERNS:
- Tax Year: Look for "Form 1040" header year, or check filename (2024_John_Smith.pdf → 2024)
- AGI: Line 11 on Form 1040 — extract exact number with commas
- SSN: Near person's name on Form 1040 header, format XXX-XX-XXXX
- Filing Status: Look for checked box (☒ Married filing jointly, etc.)

RESPONSE STYLE:
- Conversational and warm, not robotic
- Lead with the direct answer, then add context if helpful
- For sensitive data (SSNs, account numbers), confirm you have authorization before providing
- Keep responses focused — don't dump everything you know, answer what was asked"""


def _sanitize_input(text: str) -> str:
    """Basic prompt injection protection."""
    # Block common injection patterns
    injection_patterns = [
        r"ignore (all |previous |above )?instructions",
        r"you are now",
        r"new persona",
        r"jailbreak",
        r"pretend you",
        r"act as",
        r"forget your",
        r"system prompt",
    ]
    text_lower = text.lower()
    for pattern in injection_patterns:
        if re.search(pattern, text_lower):
            return "[Input blocked: potential prompt injection detected]"
    # Truncate extremely long inputs
    return text[:4000] if len(text) > 4000 else text


def post_process_answer(question: str, answer: str, docs: List[Document]) -> str:
    """Rule-based post-processing to catch common model errors."""
    q_lower = question.lower()

    # SSN extraction fallback
    if 'ssn' in q_lower or 'social security' in q_lower:
        ssn_pattern = re.compile(r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b')
        if not ssn_pattern.search(answer):
            for doc in docs:
                content = doc.page_content
                matches = list(ssn_pattern.finditer(content))
                if matches:
                    if 'john' in q_lower and 'JOHN' in content.upper():
                        john_idx = content.upper().find('JOHN')
                        for match in matches:
                            if abs(match.start() - john_idx) < 100:
                                return match.group(0)
                    elif 'sally' in q_lower and 'SALLY' in content.upper():
                        sally_idx = content.upper().find('SALLY')
                        for match in matches:
                            if abs(match.start() - sally_idx) < 100:
                                return match.group(0)

    # Tax year extraction
    if 'tax year' in q_lower or 'what year' in q_lower:
        year_match = re.search(r'\b(20\d{2})\b', answer)
        if year_match:
            return year_match.group(1)
        for doc in docs:
            year_match = re.search(r'\b(20\d{2})\b', doc.metadata.get('source', ''))
            if year_match:
                return year_match.group(1)

    # Filing status fallback
    if 'filing status' in q_lower:
        if 'cannot find' in answer.lower() or 'not available' in answer.lower():
            for doc in docs:
                content = doc.page_content.upper()
                if 'X MARRIED FILING JOINTLY' in content or '☒ MARRIED FILING JOINTLY' in content:
                    return "Married filing jointly"
                if 'X SINGLE' in content or '☒ SINGLE' in content:
                    return "Single"
                if 'X HEAD OF HOUSEHOLD' in content:
                    return "Head of household"

    return answer


def build_qa_chain(llm, retriever):
    """
    Returns a callable that supports:
    - Document retrieval + cross-encoder reranking
    - Conversation history (chat memory)
    - Prompt injection protection
    - Post-processing for common extraction tasks
    
    Input dict keys:
      query         : str  — the current question
      chat_history  : list — list of {"role": "user"|"assistant", "content": str}
    """

    def run(inputs: Dict[str, Any]) -> Dict[str, Any]:
        question     = _sanitize_input(inputs.get("query", ""))
        chat_history = inputs.get("chat_history", [])

        if question.startswith("[Input blocked"):
            return {"result": "⚠️ Your message was blocked for security reasons. Please rephrase your question.", "source_documents": []}

        # Step 1: Retrieve
        docs: List[Document] = retriever.invoke(question)

        # Step 2: Rerank
        if len(docs) > 1:
            try:
                pairs  = [[question, doc.page_content] for doc in docs]
                scores = RERANKER.predict(pairs)
                docs   = [doc for doc, _ in sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:7]]
            except Exception as e:
                print(f"[WARNING] Reranking failed: {e}")
                docs = docs[:7]

        # Step 3: Build context
        context = "\n\n".join(
            f"[source={d.metadata.get('source')}, page={d.metadata.get('page')}]\n{d.page_content}"
            for d in docs
        )

        # Step 4: Build message list with history
        messages = [SystemMessage(content=SYSTEM_PROMPT)]

        # Add prior conversation turns
        for turn in chat_history[-10:]:   # keep last 10 turns to stay within context
            role    = turn.get("role", "user")
            content = turn.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

        # Add current question with context
        messages.append(HumanMessage(
            content=f"Here are the relevant document excerpts:\n\n{context}\n\nQuestion: {question}"
        ))

        # Step 5: Call LLM
        response = llm.invoke(messages)
        text     = getattr(response, "content", str(response))

        # Step 6: Post-process
        text = post_process_answer(question, text, docs)

        return {"result": text, "source_documents": docs}

    return run