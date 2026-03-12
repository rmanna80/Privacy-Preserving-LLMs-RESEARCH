# ai_core/interactive_session.py

import time

from ai_core import FinancialQASystem
from ai_core.privacy_policy import DisclosureMode
from ai_core.audit_logger import AuditLogger

import os


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

    disclosure_mode = DisclosureMode.AUTHORIZED
    include_sources = True
    authorized = False 

    print("\nSystem ready.")
    print("Type a question and press Enter.")
    print("Commands:")
    print("  /exit")
    print("  /mode open|authorized|redacted")
    print("  /sources on|off\n")

    # user_authorized = False
    # SECURE_PASSPHRASE = os.getenv("FINQA_PASSPHRASE", "letmein") # demo only 
    print("  /login <passphrase>")
    print("  /logout")

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
            if len(parts) == 2 and parts[1].lower() in ("on", "off"):
                include_sources = (parts[1].lower() == "on")
                print(f"Sources: {'on' if include_sources else 'off'}")
            else:
                print("Usage: /sources on|off")
            continue

        if q.lower().startswith("/login"):
            parts = q.split(maxsplit=1)
            if len(parts) == 2 and parts[1] == "letmein":
                authorized = True
                print("Secure mode: ON (authorized)")
            else:
                print("Login failed.")
            continue

        if q.lower().startswith("/logout"):
            authorized = False
            print("Secure mode: OFF")
            continue
        
        # ask
        t0 = time.perf_counter()
        try:
            answer = system.ask(
                q,
                disclosure_mode=disclosure_mode,
                include_sources=include_sources,
                authorized=authorized,   # ✅ pass through
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
            logger.log({
                "question": q,
                "disclosure_mode": disclosure_mode.value,
                "authorized": authorized,
                "grounded": False,
                "sources": [],
                "latency_ms": latency_ms,
                "ui": "cli",
                "error": str(e),
            })
