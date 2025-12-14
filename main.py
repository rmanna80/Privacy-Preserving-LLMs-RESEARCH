from ai_core import FinancialQASystem


def main():
    system = FinancialQASystem(
        docs_dir="data/raw_pdfs",
        chunk_size=1000,
        chunk_overlap=150,
        verbose=True,
    )

    system.index_documents()
    system.preview_chunk(0)


if __name__ == "__main__":
    main()
