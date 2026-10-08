"""
=============================================================================
Module: evaluation/eval_failover.py
Description: Evaluates the multi-provider failover shield by forcing HTTP 429
             (ResourceExhausted) on the primary model across 15 trials.
             Verifies that the candidate loop cascades to the next available model,
             measures takeover rates, and calculates extra failover delay.
=============================================================================
"""

import sys
import time
import json
import argparse
from pathlib import Path
import pandas as pd

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.providers import (
    load_active_keys,
    execute_with_candidate_fallback,
    generate_full_response
)

FAILOVER_TEST_PROMPTS = [
    "What are the benefits of modular software design?",
    "Explain how garbage collection works in modern runtimes.",
    "Describe the role of indexes in relational databases.",
    "What is the difference between synchronous and asynchronous code?",
    "Explain what makes a function idempotent in REST APIs.",
    "How does TLS certificate verification work?",
    "What is the purpose of containerization with Docker?",
    "Explain the concept of eventual consistency in distributed systems.",
    "How do hash tables resolve collisions?",
    "Describe the difference between horizontal and vertical scaling.",
    "What are microservices and why are they used?",
    "Explain what a dead-letter queue does in messaging systems.",
    "What is the function of an API gateway in web architectures?",
    "Describe how DNS resolution finds an IP address.",
    "Explain the purpose of continuous integration and continuous deployment."
]

def evaluate_failover(trials: int = 15, delay_sec: float = 3.0):
    eval_dir = Path(__file__).resolve().parent
    results_dir = eval_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    keys = load_active_keys()
    if not keys.get("gemini") and not keys.get("openai") and not keys.get("anthropic"):
        print("⚠️ No API keys configured in .env. Skipping failover evaluation.")
        dummy_result = {"status": "skipped", "reason": "No API keys configured in .env"}
        with open(results_dir / "failover_results.json", "w", encoding="utf-8") as f:
            json.dump(dummy_result, f, indent=2)
        return dummy_result

    prompts = FAILOVER_TEST_PROMPTS[:trials]
    primary_model = "gemini-flash-latest"

    print(f"Starting Multi-Provider Failover Benchmark: {len(prompts)} trials.")
    print(f"Primary Model: {primary_model} (Will simulate 429 ResourceExhausted exception)")

    trial_records = []
    successful_failovers = 0
    takeover_counts = {}
    extra_delays = []

    for trial_idx, prompt in enumerate(prompts, 1):
        print(f"\n[Trial {trial_idx:02d}/{len(prompts)}] Prompt: {prompt[:45]}...")

        # 1. Baseline Call (Normal execution without forced 429)
        normal_duration = 1.0
        try:
            baseline_res = execute_with_candidate_fallback(
                primary_model=primary_model,
                active_keys=keys,
                prompt=prompt,
                mock_force_primary_429=False
            )
            normal_duration = baseline_res["total_elapsed"]
            print(f"  [OK] Normal Baseline : {normal_duration:.2f}s via {baseline_res['successful_model']}")
        except Exception as e:
            print(f"  [WARN] Normal Baseline error: {str(e)[:60]}")

        time.sleep(delay_sec)

        # 2. Failover Call (Force 429 on Primary Model)
        failover_res = execute_with_candidate_fallback(
            primary_model=primary_model,
            active_keys=keys,
            prompt=prompt,
            mock_force_primary_429=True
        )

        completed = failover_res["success"]
        takeover_model = failover_res["successful_model"]
        failover_duration = failover_res["total_elapsed"]
        extra_delay = max(0.0, failover_duration - normal_duration)

        if completed and failover_res["failover_occurred"]:
            successful_failovers += 1
            takeover_counts[takeover_model] = takeover_counts.get(takeover_model, 0) + 1
            extra_delays.append(extra_delay)
            print(f"  [FAILOVER OK] Rescued by {takeover_model} in {failover_duration:.2f}s (Extra delay: +{extra_delay:.2f}s)")
        else:
            print(f"  [FAILOVER FAILED] No candidate succeeded.")

        trial_records.append({
            "trial": trial_idx,
            "prompt": prompt,
            "normal_duration_sec": round(normal_duration, 2),
            "failover_duration_sec": round(failover_duration, 2),
            "extra_delay_sec": round(extra_delay, 2),
            "success": completed,
            "takeover_model": takeover_model,
            "attempts": failover_res["attempts"]
        })

        time.sleep(delay_sec)

    total_trials = len(prompts)
    success_rate = (successful_failovers / total_trials) if total_trials > 0 else 0.0
    avg_extra_delay = (sum(extra_delays) / len(extra_delays)) if extra_delays else 0.0

    metrics = {
        "total_trials": total_trials,
        "successful_failovers": successful_failovers,
        "failover_success_rate": round(success_rate, 4),
        "takeover_distribution": takeover_counts,
        "mean_extra_delay_sec": round(avg_extra_delay, 3),
        "trials_detail": trial_records
    }

    out_file = results_dir / "failover_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n==================================================================")
    print(f"Failover benchmark complete! Saved to {out_file.name}")
    print(f"Success Rate    : {success_rate:.2%}")
    print(f"Takeover Models : {takeover_counts}")
    print(f"Mean Extra Delay: +{avg_extra_delay:.2f}s")
    print(f"==================================================================")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Failover evaluation benchmark")
    parser.add_argument("--trials", type=int, default=15, help="Number of trials (default 15)")
    parser.add_argument("--delay", type=float, default=3.0, help="Delay between trials in seconds")
    args = parser.parse_args()
    evaluate_failover(trials=args.trials, delay_sec=args.delay)
