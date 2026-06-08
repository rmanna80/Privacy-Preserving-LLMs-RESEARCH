from __future__ import annotations

import argparse

from ai_core import FinancialQASystem
from ai_core.interactive_session import run_interactive_session
from ai_core.privacy_policy import DisclosureMode


def run_demo(force_rebuild: bool = False) -> None:
    """
    Demo mode for quickly testing the system on sample questions.
    """
    system = FinancialQASystem(
        docs_dir="data/raw_pdfs",
        db_dir="vectorstore/chroma_db",
        chunk_size=800,
        chunk_overlap=120,
        verbose=True,
    )
    system.index_documents(force_rebuild=force_rebuild)

    questions = [
        "What tax year is this return for?",
        "What is the filing status on this return?",
        "What is John's Social Security Number (SSN)?",
        "What is Sally's Social Security Number (SSN)?",
    ]

    for question in questions:
        print("=" * 80)
        print(f"Question: {question}\n")
        answer = system.ask(
            question,
            disclosure_mode=DisclosureMode.AUTHORIZED,
            authorized=True,
            include_sources=True,
        )
        print(answer)
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Privacy-Preserving Financial Document QA"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run sample demo questions instead of the interactive session.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force a rebuild of the vector store before running.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.demo:
        run_demo(force_rebuild=args.rebuild)
    else:
        run_interactive_session(force_rebuild=args.rebuild)


if __name__ == "__main__":
    main()