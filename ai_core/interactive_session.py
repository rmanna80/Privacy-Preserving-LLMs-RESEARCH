# ai_core/interactive_session.py

import time

from ai_core import FinancialQASystem
from ai_core.privacy_policy import DisclosureMode
from ai_core.audit_logger import AuditLogger


def run_interactive_session():
    system = FinancialQASystem(
        docs_dir="data/raw_pdfs",
        db_dir="vectorstore/chroma_db",
        chunk_size=800,
        chunk_overlap=120,
        verbose=True,
    )

    system.index_documents(force_rebuild=False)

    logger = AuditLogger("logs/audit.jsonl")

    print("\nSystem ready.")
    print("Type a question and press Enter.")
    print("Commands:")
    print("  /exit")
    print("  /mode open|authorized|redacted")
    print("  /sources on|off\n")

    disclosure_mode = DisclosureMode.AUTHORIZED
    include_sources = True

    while True:
        q = input("> ").strip()
        if not q:
            continue

        if q.lower() in ("/exit", "exit", "quit", "/quit"):
            break

        if q.lower().startswith("/mode"):
            parts = q.split()
            if len(parts) == 2:
                mode = parts[1].lower()
                if mode == "open":
                    disclosure_mode = DisclosureMode.OPEN
                elif mode == "authorized":
                    disclosure_mode = DisclosureMode.AUTHORIZED
                elif mode == "redacted":
                    disclosure_mode = DisclosureMode.REDACTED
                else:
                    print("Unknown mode.")
                    continue
                print(f"Mode set to: {disclosure_mode.value}")
            else:
                print("Usage: /mode open|authorized|redacted")
            continue

        if q.lower().startswith("/sources"):
            parts = q.split()
            if len(parts) != 2:
                print("Usage: /sources on|off")
                continue

            val = parts[1].lower()
            if val in ("on", "true", "yes"):
                include_sources = True
            elif val in ("off", "false", "no"):
                include_sources = False
            else:
                print("Usage: /sources on|off")
                continue

            print(f"Sources: {'on' if include_sources else 'off'}")
            continue

        # ask
        t0 = time.perf_counter()
        try:
            answer = system.ask(
                q,
                disclosure_mode=disclosure_mode,
                include_sources=include_sources,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            print("\n" + answer + "\n")

            trace = getattr(system, "last_trace", {}).copy()
            trace["latency_ms"] = latency_ms
            trace["ui"] = "cli"

            logger.log(trace)

        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            print(f"Error: {e}")

            logger.log(
                {
                    "question": q,
                    "disclosure_mode": disclosure_mode.value,
                    "grounded": False,
                    "sources": [],
                    "latency_ms": latency_ms,
                    "ui": "cli",
                    "error": str(e),
                }
            )
