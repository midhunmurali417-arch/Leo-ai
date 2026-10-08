# 🦁 Leo AI Assistant - Evaluation Suite Guide

This directory contains the automated evaluation and benchmarking suite for **Leo AI Assistant**. It evaluates **Intent Detection**, **Real-Time Grounding Accuracy**, **Streaming Latency**, and **Multi-Provider Failover Fault Tolerance**.

---

## 📐 Evaluation Metrics Explained (2 Lines Each)

### 1. Intent Detection Metrics
- **Accuracy:** The ratio of correct decisions (live search triggered vs. static query skipped) over all 50 test queries. It measures overall classification soundness.
- **Precision:** The proportion of queries triggered for web search that genuinely needed live web data. High precision prevents unnecessary search queries on static code, math, or definitions.
- **Recall:** The proportion of live, time-sensitive queries that were successfully detected by the intent engine. It measures how effectively Leo spots questions needing current facts.
- **F1 Score:** The harmonic mean of precision and recall. It balances false alarms with missed triggers into a single unified performance score.

### 2. Grounding Accuracy Metrics
- **Ungrounded Accuracy:** The factual correctness percentage of answers produced by the raw, static LLM without external search context. It establishes the baseline hallucination rate.
- **Grounded Accuracy:** The factual correctness percentage of answers produced when real-time Google Search and Google News context are injected. It quantifies the accuracy boost from web grounding.
- **Citation Precision:** The fraction of retrieved source citations and links that directly relate to and support the user's question. It measures citation quality and relevance.

### 3. Latency Metrics
- **Time-To-First-Token (TTFT):** The time elapsed between sending the prompt and receiving the very first streaming token. It measures immediate user responsiveness.
- **Total Latency (Mean, Median, p95):** The total time required to generate the complete response across conditions (with vs. without search). p95 captures worst-case tail latencies.

### 4. Fault Tolerance (Failover) Metrics
- **Failover Success Rate:** The percentage of queries that successfully complete when the primary model encounters an HTTP 429 rate limit or quota exhaustion. It validates system resilience.
- **Mean Extra Delay:** The extra time in seconds incurred when cascading from a failed primary model to the backup candidate. It measures failover overhead.

---

## 🚀 How to Run the Evaluations

### Step 1: Install Evaluation Dependencies
Evaluation requires `pandas` and `matplotlib` (isolated from the main app dependencies):
```powershell
pip install -r evaluation/requirements-eval.txt
```

### Step 2: Run Individual Benchmarks or Master Suite

#### Option A: Quick Full Run (All Benchmarks)
Run the master runner to execute all benchmarks in sequence and compile `evaluation/results/summary.md`:
```powershell
# Rapid run with reduced sample count (ideal for testing)
python evaluation/run_all.py --quick

# Full rigorous run (20 latency queries, 15 failover trials)
python evaluation/run_all.py
```

#### Option B: Run Specific Benchmark Scripts
1. **Evaluate Intent Detection:**
   ```powershell
   python evaluation/eval_intent.py
   ```
   *Generates:* `intent_results.json`, `intent_errors.csv`, and `intent_confusion_matrix.png`.

2. **Evaluate Grounding (Phase 1 - Run Queries):**
   ```powershell
   python evaluation/eval_grounding.py --phase run
   ```
   *Generates:* `evaluation/results/grounding_answers.csv` containing side-by-side ungrounded vs. grounded answers and citations.

3. **Score Grounding (Phase 2 - After Hand-Marking):**
   ```powershell
   python evaluation/eval_grounding.py --phase score
   ```
   *Generates:* `grounding_results.json` and `grounding_accuracy.png`.

4. **Evaluate Latency:**
   ```powershell
   python evaluation/eval_latency.py --count 20
   ```
   *Generates:* `latency_results.json` and `latency.png`.

5. **Evaluate Multi-Provider Failover:**
   ```powershell
   python evaluation/eval_failover.py --trials 15
   ```
   *Generates:* `failover_results.json`.

---

## ✍️ How to Hand-Mark `grounding_answers.csv`

To evaluate factual correctness without relying on hallucinated automated self-grading:

1. Open `evaluation/results/grounding_answers.csv` in **Excel**, **Google Sheets**, or any text editor.
2. Review each row:
   - Compare `ungrounded_answer` against the verified `reference_answer`.
   - In column **`correct_ungrounded`**, enter:
     - `1` if the ungrounded answer is factually correct.
     - `0` if the ungrounded answer is wrong, outdated, or hallucinated.
   - Compare `grounded_answer` against the verified `reference_answer`.
   - In column **`correct_grounded`**, enter:
     - `1` if the grounded answer is factually correct and up-to-date.
     - `0` if the grounded answer is incorrect.
   - In column **`citation_relevant`**, enter:
     - `1` if the retrieved sources (`citation_sources`) accurately reference the question subject.
     - `0` if the citations are irrelevant.
3. Save the CSV file.
4. Run the scoring phase:
   ```powershell
   python evaluation/eval_grounding.py --phase score
   ```
   This immediately computes ungrounded vs. grounded accuracy, citation precision, and saves `evaluation/results/grounding_accuracy.png` for your PPT presentation!
