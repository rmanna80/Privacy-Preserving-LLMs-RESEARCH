from ai_core import FinancialQASystem
from ai_core.privacy_policy import DisclosureMode
from ai_core.interactive_session import run_interactive_session

def main():
    system = FinancialQASystem(
        docs_dir="data/raw_pdfs",
        db_dir="vectorstore/chroma_db",
        chunk_size=800,
        chunk_overlap=120,
        verbose=True,
    )

    system.index_documents(force_rebuild=False)

    ''' this is for debugging, the .search() method'''
    # try a couple retrieval queries"
    # system.search("What is the filling status on this return?", k=4)
    # system.search("What is the social security number listed?", k=4)
    
    # Example questions
    questions = [
        "What tax year is this return for?",
        "What is the filing status on this return?",
        "What is John's Social Security Number (SSN)?",
    ]

    for q in questions:
        print("=" * 80)
        print(f"Question: {q}\n")

        answer = system.ask(
            q,
            disclosure_mode=DisclosureMode.AUTHORIZED,
            include_sources=True,
        )

        print(answer)
        print()


if __name__ == "__main__":
    run_interactive_session()
