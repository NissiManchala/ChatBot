"""Benchmark queries against the running API.

1. Start the API:   uvicorn app:app --reload
2. Run this script: python tests_sample_queries.py [--url http://127.0.0.1:8000/chat]

In-scope queries should get a grounded answer with confidence > 0; the last query is a
validation test and must be refused.
"""
import argparse
import sys

import requests

REFUSAL_MARKER = "couldn't find this"

# (query, should_be_answered)
QUERIES = [
    ("What is Agentic AI according to the eBook?", True),
    ("How do AI agents differ from traditional automation systems?", True),
    ("What are the core components of an Agentic Architecture?", True),
    ("What role does memory play in Agentic AI workflows?", True),
    ("What are the readiness levels for adopting Agentic AI?", True),
    ("Who won the 2022 FIFA World Cup?", False),  # out of scope: must refuse
]


def main(url: str) -> int:
    failures = 0
    for i, (query, should_answer) in enumerate(QUERIES, 1):
        print(f"\n{'=' * 78}\nQ{i}: {query}")
        try:
            resp = requests.post(url, json={"query": query}, timeout=90)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  ERROR: {e}")
            failures += 1
            continue

        answered = REFUSAL_MARKER not in data["answer"].lower() and data["confidence_score"] > 0
        ok = answered == should_answer
        failures += 0 if ok else 1

        print(f"A: {data['answer']}")
        print(f"Confidence: {data['confidence_score']}")
        for c in data["retrieved_chunks"]:
            snippet = c["text"].replace("\n", " ")[:110]
            print(f"  - p.{c['page']} sim={c['score']:.3f}  {snippet}...")
        print(f"Result: {'PASS' if ok else 'FAIL'} "
              f"(expected {'an answer' if should_answer else 'a refusal'})")

    print(f"\n{'=' * 78}\n{len(QUERIES) - failures}/{len(QUERIES)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/chat")
    sys.exit(main(parser.parse_args().url))
