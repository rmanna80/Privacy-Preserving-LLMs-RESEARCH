# ai_core/qa_chain.py

from __future__ import annotations

from typing import Dict, Any, List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


SYSTEM_PROMPT = (
    "You are a careful financial document assistant. "
    "Answer using ONLY the provided context. "
    "If the answer is not in the context, say you cannot find it."
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
    ]
)


def build_qa_chain(llm, retriever):
    """
    Returns a callable that:
    - retrieves documents
    - formats context
    - calls the LLM
    - returns {"result": ..., "source_documents": [...]}
    """

    def run(inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs["query"]

        # retrieve docs
        docs: List[Document] = retriever.invoke(question)

        # combine context
        context = "\n\n".join(
            f"[source={d.metadata.get('source')}, page={d.metadata.get('page')}]\n{d.page_content}"
            for d in docs
        )

        # call LLM
        messages = PROMPT.format_messages(context=context, question=question)
        response = llm.invoke(messages)

        # ChatOllama returns an AIMessage-like object
        text = getattr(response, "content", str(response))

        return {"result": text, "source_documents": docs}

    return run
