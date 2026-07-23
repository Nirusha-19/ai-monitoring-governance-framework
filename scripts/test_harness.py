"""
Simulates realistic traffic against the running FastAPI /ask endpoint, normal questions, out-of-scope questions, and edge cases, covering
every decision path the framework can produce.
"""
import requests
import time

API_URL = "http://127.0.0.1:8000/ask"

TEST_QUESTIONS = {
    "normal": [
        "How do I reset my email password?",
        "My VPN keeps disconnecting, how do I fix it?",
        "How do I connect a printer to my laptop?",
        "My computer won't turn on, what should I check?",
    ],
    "out_of_scope": [
        "What's the best recipe for chocolate cake?",
        "What's the weather like in Paris tomorrow?",
        "Who won the World Cup in 2022?",
    ],
    "edge_case": [
        "",  # empty question
        "help",  # extremely vague
        "asdkjfh alsdkjf laksjdf",  # gibberish
    ],
}


def run_test(question: str, category: str):
    start = time.time()
    try:
        response = requests.post(API_URL, json={"question": question}, timeout=60)
        elapsed = time.time() - start
        if response.status_code == 200:
            data = response.json()
            return {
                "category": category,
                "question": question,
                "status": "success",
                "decision": data["decision"],
                "failed_checks": data["failed_checks"],
                "rerank_score": data["rerank_score"],
                "is_refusal": data["is_refusal"],
                "http_latency_s": round(elapsed, 1),
            }
        else:
            return {
                "category": category,
                "question": question,
                "status": f"http_error_{response.status_code}",
                "http_latency_s": round(elapsed, 1),
            }
    except requests.exceptions.RequestException as e:
        return {
            "category": category,
            "question": question,
            "status": f"request_failed: {e}",
            "http_latency_s": round(time.time() - start, 1),
        }


def main():
    print(f"Sending test traffic to {API_URL}...")
    print("(Make sure the server is running in another terminal.)\n")

    results = []
    for category, questions in TEST_QUESTIONS.items():
        print(f"\n--- {category.upper()} ---")
        for q in questions:
            result = run_test(q, category)
            results.append(result)
            q_display = q if q else "(empty string)"
            print(f"  [{result['status']}] '{q_display[:50]}' "
                  f"-> decision={result.get('decision', 'N/A')}, "
                  f"latency={result['http_latency_s']}s")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    successful = [r for r in results if r["status"] == "success"]
    print(f"Total requests: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed/errored: {len(results) - len(successful)}")

    if successful:
        flagged = [r for r in successful if r["decision"] == "flagged_for_review"]
        print(f"\nFlagged for review: {len(flagged)}/{len(successful)}")
        for category in TEST_QUESTIONS:
            cat_results = [r for r in successful if r["category"] == category]
            cat_flagged = [r for r in cat_results if r["decision"] == "flagged_for_review"]
            if cat_results:
                print(f"  {category}: {len(cat_flagged)}/{len(cat_results)} flagged")


if __name__ == "__main__":
    main()
