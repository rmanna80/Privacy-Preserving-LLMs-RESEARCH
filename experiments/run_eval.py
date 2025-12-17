# experiments/run_eval.py

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import json
import time

from ai_core import FinancialQASystem
from ai_core.privacy_policy import DisclosureMode
from ai_core.audit_logger import AuditLogger

def load_questions(path: str | Path):
    """
    Supports:
    - .txt: one question per line (blank lines ignored)
    - .json: {"questions": ["...", "..."]}
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")

    if path.suffix.lower() == ".txt":
        qs = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                qs.append(line)
        return qs

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("questions", [])

    raise ValueError("Unsupported questions file. Use .txt or .json")


def main():
    # ---- config ----
    docs_dir = "data/raw_pdfs"
    db_dir = "vectorstore/chroma_db"
    questions_file = "experiments/questions.txt"  # create this file
    output_log = "logs/eval.jsonl"

    disclosure_mode = DisclosureMode.AUTHORIZED
    include_sources = True
    force_rebuild = False

    # ---- init system ----
    system = FinancialQASystem(
        docs_dir=docs_dir,
        db_dir=db_dir,
        chunk_size=800,
        chunk_overlap=120,
        verbose=True,
    )
    system.index_documents(force_rebuild=force_rebuild)

    logger = AuditLogger(output_log)
    questions = load_questions(questions_file)

    print(f"\nLoaded {len(questions)} questions from {questions_file}")
    print(f"Logging to {output_log}\n")

    # ---- run ----
    for i, q in enumerate(questions, start=1):
        t0 = time.perf_counter()
        try:
            _ = system.ask(
                q,
                disclosure_mode=disclosure_mode,
                include_sources=include_sources,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0

            trace = getattr(system, "last_trace", {}).copy()
            trace["latency_ms"] = latency_ms
            trace["ui"] = "eval"
            trace["question_index"] = i
            trace["questions_file"] = str(questions_file)
            logger.log(trace)

            print(f"[{i}/{len(questions)}] ok ({latency_ms:.1f} ms): {q}")

        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            logger.log(
                {
                    "question": q,
                    "disclosure_mode": disclosure_mode.value,
                    "grounded": False,
                    "sources": [],
                    "latency_ms": latency_ms,
                    "ui": "eval",
                    "question_index": i,
                    "questions_file": str(questions_file),
                    "error": str(e),
                }
            )
            print(f"[{i}/{len(questions)}] ERROR: {q} -> {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
