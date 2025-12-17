# ai_core/llm_ollama.py

from langchain_ollama import ChatOllama


def build_ollama_llm(model: str = "llama3.1:8b", temperature: float = 0.2):
    return ChatOllama(model=model, temperature=temperature)