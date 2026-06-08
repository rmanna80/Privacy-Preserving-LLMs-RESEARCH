from __future__ import annotations

import os
import time
from typing import Optional, Tuple

from ai_core import FinancialQASystem
from ai_core.audit_logger import AuditLogger
from ai_core.privacy_policy import DisclosureMode


DEFAULT_PASSPHRASE = "letmein"


def _build_system(force_rebuild: bool = False) -> FinancialQASystem:
    system = FinancialQASystem(
        docs_dir="data/raw_pdfs",
        db_dir="vectorstore/chroma_db",
        chunk_size=800,
        chunk_overlap=120,
        verbose=True,
    )
    system.index_documents(force_rebuild=force_rebuild)
    return system


def _print_banner() -> None:
    print("\nSystem ready.")
    print("Type a question and press Enter.")
    print("Commands:")
    print(" /exit")
    print(" /mode open|authorized|redacted")
    print(" /sources on|off")
    print(" /login <passphrase>")
    print(" /logout")
    print(" /help\n")


def _handle_mode_command(
    command: str,
    current_mode: DisclosureMode,
) -> Tuple[DisclosureMode, bool]:
    parts = command.split()
    if len(parts) != 2:
        print("Usage: /mode open|authorized|redacted")
        return current_mode, True

    mode = parts[1].lower()
    if mode == "open":
        current_mode = DisclosureMode.OPEN
    elif mode == "authorized":
        current_mode = DisclosureMode.AUTHORIZED
    elif mode == "redacted":
        current_mode = DisclosureMode.REDACTED
    else:
        print("Unknown mode.")
        return current_mode, True

    print(f"Mode set to: {current_mode.value}")
    return current_mode, True


def _handle_sources_command(command: str, include_sources: bool) -> Tuple[bool, bool]:
    parts = command.split()
    if len(parts) != 2 or parts[1].lower() not in ("on", "off"):
        print("Usage: /sources on|off")
        return include_sources, True

    include_sources = parts[1].lower() == "on"
    print(f"Sources: {'on' if include_sources else 'off'}")
    return include_sources, True


def _handle_login_command(command: str, authorized: bool) -> Tuple[bool, bool]:
    parts = command.split(maxsplit=1)
    expected = os.getenv("FINQA_PASSPHRASE", DEFAULT_PASSPHRASE)

    if len(parts) == 2 and parts[1] == expected:
        authorized = True
        print("Secure mode: ON (authorized)")
    else:
        print("Login failed.")

    return authorized, True


def _handle_special_command(
    command: str,
    disclosure_mode: DisclosureMode,
    include_sources: bool,
    authorized: bool,
) -> Tuple[DisclosureMode, bool, bool, bool]:
    """
    Returns:
    - updated disclosure_mode
    - updated include_sources
    - updated authorized
    - handled flag
    """
    cmd = command.lower().strip()

    if cmd in ("/help", "help"):
        _print_banner()
        return disclosure_mode, include_sources, authorized, True

    if cmd.startswith("/mode"):
        disclosure_mode, handled = _handle_mode_command(command, disclosure_mode)
        return disclosure_mode, include_sources, authorized, handled

    if cmd.startswith("/sources"):
        include_sources, handled = _handle_sources_command(command, include_sources)
        return disclosure_mode, include_sources, authorized, handled

    if cmd.startswith("/login"):
        authorized, handled = _handle_login_command(command, authorized)
        return disclosure_mode, include_sources, authorized, handled

    if cmd.startswith("/logout"):
        authorized = False
        print("Secure mode: OFF")
        return disclosure_mode, include_sources, authorized, True

    return disclosure_mode, include_sources, authorized, False


def run_interactive_session(force_rebuild: bool = False) -> None:
    system = _build_system(force_rebuild=force_rebuild)
    logger = AuditLogger("logs/audit.jsonl")

    disclosure_mode = DisclosureMode.AUTHORIZED
    include_sources = True
    authorized = False

    _print_banner()

    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not q:
            continue

        if q.lower() in ("/exit", "exit", "quit", "/quit"):
            break

        disclosure_mode, include_sources, authorized, handled = _handle_special_command(
            q,
            disclosure_mode,
            include_sources,
            authorized,
        )
        if handled:
            continue

        t0 = time.perf_counter()

        try:
            answer = system.ask(
                q,
                disclosure_mode=disclosure_mode,
                include_sources=include_sources,
                authorized=authorized,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0

            print("\n" + answer + "\n")

            trace = getattr(system, "last_trace", {}).copy()
            trace["latency_ms"] = latency_ms
            trace["ui"] = "cli"
            logger.log(trace)

        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            print(f"Error: {exc}")

            logger.log(
                {
                    "question": q,
                    "disclosure_mode": disclosure_mode.value,
                    "authorized": authorized,
                    "grounded": False,
                    "sources": [],
                    "latency_ms": latency_ms,
                    "ui": "cli",
                    "error": str(exc),
                }
            )