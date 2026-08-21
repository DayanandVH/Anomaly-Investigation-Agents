"""
Evaluation harness for the anomaly-investigation agent.

Two things are checked, since they fail in different ways and need
different tests:

1. RETRIEVAL ACCURACY — does the knowledge base return the pattern you'd
   expect for a given anomaly shape? This tests knowledge_base.py in
   isolation, with no LLM involved.

2. EXPLANATION GROUNDEDNESS — does the LLM's explanation actually reference
   the real feature names it was given, rather than hallucinating or
   drifting from the numbers? This is exactly the kind of check that would
   have caught the z-score range mistake seen during manual testing previously:
   the model claimed a value was "in the 1.5-2.5 range" when the real value
   was 2.79. An automated groundedness check surfaces that kind of slip
   instead of relying on a human happening to notice it.

Run with:
    python src/evaluate.py            (uses Groq — needs GROQ_API_KEY set)
    python src/evaluate.py --offline  (retrieval-only, no LLM calls)
"""

import re
import sys
from dataclasses import asdict
from typing import List, Dict

from data import load_data
from detector import AnomalyDetector
from knowledge_base import build_knowledge_base, retrieve as kb_retrieve
from agent import build_agent, describe_anomaly_shape


# --- Part 1: retrieval accuracy ---
# Each case is a synthetic AnomalyFlag-shaped dict paired with the pattern ID
# we EXPECT the knowledge base to return. Building the query via
# describe_anomaly_shape() (the SAME function agent.py actually uses) means
# this test can never silently drift from what production really sends —
# if that function changes, this eval automatically tests the new behavior.
RETRIEVAL_EVAL_CASES = [
    {
        "anomaly": {"top_features": [{"feature": "V9", "z_score": 6.1, "value": 4.2},
                                      {"feature": "V4", "z_score": 5.0, "value": 3.9},
                                      {"feature": "V6", "z_score": 4.5, "value": 3.1}],
                    "raw_row": {"Amount": 1.20}},
        "expected_id": "pattern_card_testing",
    },
    {
        "anomaly": {"top_features": [{"feature": "V4", "z_score": 4.8, "value": -6.0}],
                    "raw_row": {"Amount": 920.00}},
        "expected_id": "pattern_single_large_anomaly",
    },
    {
        "anomaly": {"top_features": [{"feature": f"V{i}", "z_score": 2.1, "value": 1.0} for i in range(1, 8)],
                    "raw_row": {"Amount": 60.00}},
        "expected_id": "pattern_broad_mild_deviation",
    },
]


def evaluate_retrieval(kb, cases: List[Dict] = RETRIEVAL_EVAL_CASES) -> float:
    """Return the fraction of cases where the top-1 retrieved document matches
    the expected pattern ID. Also prints per-case results for inspection.
    """
    correct = 0
    for case in cases:
        query = describe_anomaly_shape(case["anomaly"])
        results = kb.query(query_texts=[query], n_results=1)
        top_id = results["ids"][0][0]
        is_correct = top_id == case["expected_id"]
        correct += is_correct
        status = "PASS" if is_correct else "FAIL"
        print(f"  [{status}] query=\"{query}\"")
        print(f"           expected={case['expected_id']:30s} got={top_id}")
    accuracy = correct / len(cases)
    print(f"\nRetrieval accuracy: {correct}/{len(cases)} ({accuracy:.0%})")
    return accuracy


# --- Part 2: explanation groundedness ---

def check_groundedness(anomaly: Dict, explanation: str) -> Dict:
    """Check whether an explanation actually references the real feature
    names it was given. This is a simple, cheap check (not a full factual
    audit) — it catches the most common and most damaging failure mode:
    the LLM talking about features that were never actually flagged.

    Returns a dict with the fraction of top features mentioned, and which
    (if any) were missing.
    """
    feature_names = [f["feature"] for f in anomaly["top_features"]]
    # IGNORECASE matters here: the LLM naturally writes "amount" lowercase in
    # prose even though the column is named "Amount" — an earlier version of
    # this check was case-sensitive and wrongly flagged every correct mention
    # of amount as "missing".
    mentioned = [name for name in feature_names if re.search(rf"\b{re.escape(name)}\b", explanation, re.IGNORECASE)]
    missing = [name for name in feature_names if name not in mentioned]
    return {
        "feature_names": feature_names,
        "mentioned": mentioned,
        "missing": missing,
        "coverage": len(mentioned) / len(feature_names) if feature_names else 0.0,
    }


def evaluate_explanations(agent, flags, n: int = 5) -> float:
    """Run the agent on the first n flagged anomalies and report average
    feature-name groundedness across them.
    """
    total_coverage = 0.0
    for flag in flags[:n]:
        anomaly_dict = asdict(flag)
        result = agent.invoke({"anomaly": anomaly_dict, "query": "", "retrieved": [], "explanation": ""})
        check = check_groundedness(anomaly_dict, result["explanation"])
        total_coverage += check["coverage"]

        status = "PASS" if check["coverage"] == 1.0 else "PARTIAL" if check["coverage"] > 0 else "FAIL"
        print(f"  [{status}] Row {flag.row_index}: mentioned {len(check['mentioned'])}/{len(check['feature_names'])} "
              f"flagged features ({check['coverage']:.0%})")
        if check["missing"]:
            print(f"           missing: {check['missing']}")

    avg_coverage = total_coverage / n
    print(f"\nAverage feature groundedness: {avg_coverage:.0%}")
    return avg_coverage


if __name__ == "__main__":
    offline = "--offline" in sys.argv

    print("=== Retrieval evaluation ===")
    kb = build_knowledge_base(offline=offline)
    evaluate_retrieval(kb)

    if offline:
        print("\n(Skipping explanation evaluation — needs a real LLM; run without --offline)")
        sys.exit(0)

    print("\n=== Explanation groundedness evaluation ===")
    df, source = load_data()
    feature_cols = [c for c in df.columns if c != "Class"]
    detector = AnomalyDetector(contamination=0.02).fit(df, feature_cols)
    flags = detector.score(df)

    from langchain_groq import ChatGroq
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    agent = build_agent(kb, llm)

    evaluate_explanations(agent, flags, n=5)
