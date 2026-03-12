# ai_core/llm_ollama.py

# from langchain_ollama import ChatOllama


# def build_ollama_llm(model: str = "llama3.1:8b", temperature: float = 0.2):
#     return ChatOllama(model=model, temperature=temperature)


# ai_core/llm_ollama.py

from langchain_ollama import ChatOllama


def build_ollama_llm(model: str = "qwen2.5:7b", temperature: float = 0.0):  # Changed model and temp
    """
    Build Ollama LLM for financial document Q&A
    
    Recommended models:
    - qwen2.5:7b - Best for structured documents and extraction
    - llama3.2:3b - Lighter, good general performance  
    - mistral:7b - Good balance
    - llama3.1:8b - Conservative but reliable
    """
    return ChatOllama(
        model=model, 
        temperature=temperature,
        num_ctx=8192,  # Increase context window
        num_predict=512,
    )