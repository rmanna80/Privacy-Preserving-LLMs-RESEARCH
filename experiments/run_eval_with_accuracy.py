import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import json
import time
from ai_core import FinancialQASystem
from ai_core.privacy_policy import DisclosureMode
from ai_core.audit_logger import AuditLogger


def load_ground_truth(path: str | Path):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("questions", [])


def calculate_accuracy(predicted: str, expected: str) -> float:
    """Simple exact match or substring match"""
    predicted = predicted.lower().strip()
    expected = expected.lower().strip()
    
    # Exact match
    if predicted == expected:
        return 1.0
    
    # Substring match (for SSNs with different formatting)
    if expected in predicted:
        return 1.0
    
    # Check if digits match (for SSNs)
    pred_digits = ''.join(c for c in predicted if c.isdigit())
    exp_digits = ''.join(c for c in expected if c.isdigit())
    if pred_digits and pred_digits == exp_digits:
        return 1.0
    
    return 0.0


def main():
    # Config
    ground_truth_file = "experiments/ground_truth.json"
    output_log = "logs/eval_accuracy.jsonl"
    
    # Init system
    system = FinancialQASystem(
        docs_dir="data/raw_pdfs",
        db_dir="vectorstore/chroma_db",
        chunk_size=800,
        chunk_overlap=120,
        verbose=True,
    )
    system.index_documents(force_rebuild=False)
    
    logger = AuditLogger(output_log)
    test_cases = load_ground_truth(ground_truth_file)
    
    print(f"\nRunning evaluation on {len(test_cases)} test cases\n")
    
    results = []
    for i, test in enumerate(test_cases, start=1):
        question = test["question"]
        expected = test["expected_answer"]
        category = test.get("category", "unknown")
        
        t0 = time.perf_counter()
        try:
            answer = system.ask(
                question,
                disclosure_mode=DisclosureMode.AUTHORIZED,
                authorized=True,
                include_sources=False,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            
            # Calculate accuracy
            accuracy = calculate_accuracy(answer, expected)
            
            # Log result
            trace = getattr(system, "last_trace", {}).copy()
            trace.update({
                "latency_ms": latency_ms,
                "ui": "eval_accuracy",
                "question_index": i,
                "expected_answer": expected,
                "actual_answer": answer,
                "accuracy": accuracy,
                "category": category,
            })
            logger.log(trace)
            results.append(trace)
            
            status = "✓" if accuracy == 1.0 else "✗"
            print(f"[{i}/{len(test_cases)}] {status} {category}: {question}")
            print(f"  Expected: {expected}")
            print(f"  Got: {answer}")
            print(f"  Accuracy: {accuracy:.2f}\n")
            
        except Exception as e:
            print(f"[{i}/{len(test_cases)}] ERROR: {question} -> {e}\n")
    
    # Summary
    total_accuracy = sum(r["accuracy"] for r in results) / len(results) if results else 0
    print(f"\n{'='*60}")
    print(f"Overall Accuracy: {total_accuracy:.2%}")
    print(f"Correct: {sum(1 for r in results if r['accuracy'] == 1.0)}/{len(results)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()