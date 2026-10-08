"""
=============================================================================
Module: evaluation/eval_latency.py
Description: Evaluates Time-To-First-Token (TTFT) and Total Response Latency
             under two conditions: 'no_search' vs 'with_search' across all
             configured providers. Computes mean, median, and p95 percentiles.
=============================================================================
"""

import sys
import time
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import google_search
from evaluation.providers import (
    load_active_keys,
    generate_full_response,
    ALL_MODELS,
    is_rate_limit_error
)

# Standard benchmark queries for latency evaluation
BENCHMARK_QUERIES = [
    "Explain what an API is in simple terms.",
    "What is the difference between TCP and UDP?",
    "How does the binary search algorithm operate?",
    "Summarize how optical fibers transmit data.",
    "Explain the concept of database normalization.",
    "What is the function of the CPU cache?",
    "Describe the key principles of Object-Oriented Programming.",
    "What is the purpose of an operating system kernel?",
    "How does HTTPS ensure secure communication?",
    "Explain the difference between a process and a thread.",
    "What is the latest status of global renewable energy?",
    "Current developments in quantum computing hardware.",
    "Explain what is happening in autonomous vehicle technology.",
    "Recent updates in semiconductor chip manufacturing.",
    "What are the newest trends in artificial intelligence?",
    "Explain how electric vehicle battery technology works.",
    "What are the latest advancements in space exploration?",
    "Describe current innovations in cloud computing architectures.",
    "What is the role of GPU acceleration in deep learning?",
    "Summarize how modern search engines rank websites."
]

def evaluate_latency(query_count: int = 20, delay_sec: float = 3.5):
    eval_dir = Path(__file__).resolve().parent
    results_dir = eval_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    keys = load_active_keys()
    search_key = keys.get("google_search", "")
    cse_id = keys.get("google_cse", "")

    # Identify configured providers to test
    test_providers = []
    if keys.get("gemini"):
        test_providers.append(("gemini", "gemma-4-26b-a4b-it", "Google Gemini"))
    if keys.get("openai"):
        test_providers.append(("openai", "gpt-4o-mini", "GPT-4o Mini"))
    if keys.get("anthropic"):
        test_providers.append(("anthropic", "claude-3-5-haiku-20241022", "Claude 3.5 Haiku"))

    if not test_providers:
        print("⚠️ No AI provider API keys configured in .env. Skipping latency evaluation.")
        dummy_result = {"status": "skipped", "reason": "No configured provider keys in .env"}
        with open(results_dir / "latency_results.json", "w", encoding="utf-8") as f:
            json.dump(dummy_result, f, indent=2)
        return dummy_result

    queries = BENCHMARK_QUERIES[:query_count]
    print(f"Starting Latency Benchmark: {len(queries)} queries across {len(test_providers)} configured providers.")

    raw_records = []

    for prov_name, model_name, display_name in test_providers:
        api_key = keys[prov_name]
        print(f"\n--- Testing Provider: {display_name} ({prov_name}) ---")

        for condition in ["no_search", "with_search"]:
            print(f" Condition: {condition}")
            for idx, q in enumerate(queries, 1):
                if condition == "no_search":
                    effective_prompt = q
                else:
                    # Retrieve grounding context
                    s_data = google_search.execute_google_search(
                        query=q,
                        api_key=search_key,
                        cse_id=cse_id,
                        num_results=3,
                        mode="auto"
                    )
                    ctx = s_data.get("prompt_text", "")
                    effective_prompt = f"[GROUNDING CONTEXT]\n{ctx}\n\n[USER QUESTION]\n{q}" if ctx else q

                # Attempt call with 1 retry on error
                success = False
                ttft, total_time = 0.0, 0.0
                for attempt in range(2):
                    try:
                        _, ttft, total_time = generate_full_response(
                            provider=prov_name,
                            model_name=model_name,
                            api_key=api_key,
                            prompt=effective_prompt,
                            temperature=0.2
                        )
                        success = True
                        break
                    except Exception as e:
                        if attempt == 0:
                            time.sleep(2.0)
                        else:
                            print(f"   [Query {idx}] Error: {str(e)[:60]}")

                if success:
                    raw_records.append({
                        "provider": display_name,
                        "condition": condition,
                        "query_idx": idx,
                        "ttft_sec": round(ttft, 3),
                        "total_time_sec": round(total_time, 3)
                    })
                    print(f"   Query {idx:02d}: TTFT = {ttft:.3f}s | Total = {total_time:.2f}s")
                
                time.sleep(delay_sec)

    if not raw_records:
        print("No successful latency records captured.")
        return

    df = pd.DataFrame(raw_records)
    summary_results = {}

    for prov in df["provider"].unique():
        summary_results[prov] = {}
        for cond in ["no_search", "with_search"]:
            sub = df[(df["provider"] == prov) & (df["condition"] == cond)]
            if len(sub) > 0:
                summary_results[prov][cond] = {
                    "samples": len(sub),
                    "ttft": {
                        "mean": round(float(sub["ttft_sec"].mean()), 3),
                        "median": round(float(sub["ttft_sec"].median()), 3),
                        "p95": round(float(np.percentile(sub["ttft_sec"], 95)), 3),
                    },
                    "total_time": {
                        "mean": round(float(sub["total_time_sec"].mean()), 3),
                        "median": round(float(sub["total_time_sec"].median()), 3),
                        "p95": round(float(np.percentile(sub["total_time_sec"], 95)), 3),
                    }
                }

    # Save latency JSON
    out_json = results_dir / "latency_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)
    print(f"\nSaved latency metrics to {out_json.name}")

    # Plot Latency Comparison Chart
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    providers = list(summary_results.keys())
    x = np.arange(len(providers))
    width = 0.35

    # Plot 1: TTFT
    ax1 = axes[0]
    no_search_ttft = [summary_results[p].get("no_search", {}).get("ttft", {}).get("mean", 0.0) for p in providers]
    with_search_ttft = [summary_results[p].get("with_search", {}).get("ttft", {}).get("mean", 0.0) for p in providers]
    b1 = ax1.bar(x - width/2, no_search_ttft, width, label="No Search", color="#94a3b8")
    b2 = ax1.bar(x + width/2, with_search_ttft, width, label="With Grounding", color="#0284c7")
    ax1.set_title("Time-To-First-Token (TTFT)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Seconds", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(providers, fontsize=10)
    ax1.legend()

    # Plot 2: Total Time
    ax2 = axes[1]
    no_search_tot = [summary_results[p].get("no_search", {}).get("total_time", {}).get("mean", 0.0) for p in providers]
    with_search_tot = [summary_results[p].get("with_search", {}).get("total_time", {}).get("mean", 0.0) for p in providers]
    ax2.bar(x - width/2, no_search_tot, width, label="No Search", color="#94a3b8")
    ax2.bar(x + width/2, with_search_tot, width, label="With Grounding", color="#38bdf8")
    ax2.set_title("Total Generation Latency", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Seconds", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(providers, fontsize=10)
    ax2.legend()

    plt.suptitle("Latency Benchmark: No Search vs. Real-Time Grounded", fontsize=12, fontweight="bold")
    plt.tight_layout()
    chart_path = results_dir / "latency.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved latency chart to {chart_path.name}")

    return summary_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latency benchmark")
    parser.add_argument("--count", type=int, default=20, help="Number of queries per condition (default 20)")
    parser.add_argument("--delay", type=float, default=3.5, help="Delay between calls in seconds")
    args = parser.parse_args()
    evaluate_latency(query_count=args.count, delay_sec=args.delay)
