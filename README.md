# Privacy-Preserving Financial Document Analysis

A local, privacy-first AI system for analyzing sensitive financial documents using RAG and open-source LLMs.

## Features

- 🔒 **100% Local Processing** - No data leaves your machine
- 📄 **Multi-Document RAG** - Semantic search across tax returns, insurance policies, etc.
- 🛡️ **Privacy Controls** - Three-tier disclosure system (OPEN/AUTHORIZED/REDACTED)
- 🔍 **PII Extraction** - Intelligent SSN and sensitive data extraction
- 📊 **Audit Logging** - Complete query/response tracking
- 🧪 **Evaluation Framework** - Accuracy testing and performance benchmarking

## Installation
```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install Tesseract OCR (for PDF processing)
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
# Mac: brew install tesseract
# Linux: sudo apt-get install tesseract-ocr

# 3. Install Ollama and pull model
# Download from: https://ollama.ai
ollama pull llama3.1:8b
```

## Usage

### Interactive Mode
```bash
python main.py
```

### Batch Evaluation
```bash
python experiments/run_eval.py
python experiments/run_eval_with_accuracy.py
```

### Analyze Results
```bash
python analysis/analyze_logs.py --log logs/eval.jsonl --out analysis/out
```

## Project Structure

See file tree in proposal document.

## Privacy Disclosure Modes

- **OPEN**: All data returned (for testing only)
- **AUTHORIZED**: PII only shown if user authenticated + answer grounded in documents
- **REDACTED**: All PII masked with [SSN] tokens

## Research

This system is part of M.S. Business Analytics research at Elon University investigating privacy-preserving approaches to financial document AI.

**Advisor**: Dr. Emily Wang  
**Student**: Ryan Manna