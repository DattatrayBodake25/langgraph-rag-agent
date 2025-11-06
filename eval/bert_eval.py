import json
from bert_score import score
from src.main import run_agent
from src.utils import log

# File paths
RESULT_FILE = "eval/results.json"          # Initial generated answers
OUTPUT_FILE = "eval/bert_results.json"     # Final evaluation results

# Queries and reference answers
QUERIES = [
    "How does renewable energy help reduce greenhouse gas emissions?",
    "What are the benefits of solar energy?",
    "Explain the advantages of wind energy for rural areas."
]

REFERENCE_ANSWERS = [
    "Using renewable energy sources such as solar, wind, and biomass reduces CO2 and other greenhouse gases.",
    "Solar energy is sustainable, cost-effective, and environmentally friendly.",
    "Wind energy provides clean electricity, supports rural electrification, and reduces dependence on fossil fuels."
]


# Functions
def generate_results(queries, references):
    """Run agent on queries and prepare results JSON."""
    results = []
    for query, ref in zip(queries, references):
        log(f"Generating answer for query: {query}")
        agent_result = run_agent(query)
        gen_answer = agent_result.get("answer", "")
        results.append({
            "query": query,
            "generated_answer": gen_answer,
            "reference_answer": ref
        })

    # Save initial results
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    log(f"Generated answers saved to '{RESULT_FILE}'.")
    return results

def evaluate_bertscore(data):
    """Compute BERTScore between generated answers and references."""
    for entry in data:
        gen = entry.get("generated_answer", "")
        ref = entry.get("reference_answer", "")
        if not gen or not ref:
            entry["bertscore"] = {"precision": None, "recall": None, "f1": None}
            continue
        P, R, F1 = score([gen], [ref], lang="en", rescale_with_baseline=True)
        entry["bertscore"] = {
            "precision": round(P.item(), 4),
            "recall": round(R.item(), 4),
            "f1": round(F1.item(), 4)
        }
    return data

def save_results(data, file_path=OUTPUT_FILE):
    """Save final BERTScore evaluation results to JSON."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    log(f"BERTScore evaluation results saved to '{file_path}'.")


# Main Execution
if __name__ == "__main__":
    # Generate answers
    results = generate_results(QUERIES, REFERENCE_ANSWERS)

    # Evaluate with BERTScore
    evaluated_results = evaluate_bertscore(results)

    # Save final evaluation
    save_results(evaluated_results)

    log(f"Evaluation completed for {len(evaluated_results)} queries.")