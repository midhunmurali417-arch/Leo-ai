# 🦁 Leo AI Assistant - Evaluation Summary Report
*Generated At: 2026-10-08 12:10:43*

This report aggregates quantitative performance metrics for the Leo AI Assistant across Intent Detection, Grounding Accuracy, Streaming Latency, and Multi-Provider Failover.

---

## 📊 Core Performance Metrics Table

| Dimension | Evaluation Metric | Result | Notes / Context |
|---|---|---|---|
| **Intent Detection** | **Accuracy** | **86.0%** | Evaluated on 50 balanced queries (25 live vs 25 static) |
| **Intent Detection** | **Precision** | **1.0000** | Zero false-positives on static code/math/history queries |
| **Intent Detection** | **Recall** | **0.7200** | Catches temporal signals; honest evaluation on tricky queries |
| **Intent Detection** | **F1 Score** | **0.8372** | Harmonic mean of precision and recall |
| **Factual Grounding** | **Ungrounded Accuracy** | **pending** | Static LLM without external search grounding |
| **Factual Grounding** | **Grounded Accuracy** | **pending** | Real-time Google Search & News grounding injected |
| **Factual Grounding** | **Citation Precision** | **pending** | Percentage of retrieved source links verified relevant |
| **Latency Benchmark** | **Mean Latency (No Search)** | **10.20s (Google Gemini)** | Baseline direct LLM streaming generation |
| **Latency Benchmark** | **Mean Latency (With Search)** | **10.22s (Google Gemini)** | Retrieval + synthesis end-to-end latency |
| **Fault Tolerance** | **Failover Success Rate** | **100.0%** | Recovery rate under simulated HTTP 429 quota exhaustion |
| **Fault Tolerance** | **Takeover Distribution** | **gemini-flash-lite-latest: 15** | Models successfully handling cascade requests |
| **Fault Tolerance** | **Mean Extra Delay** | **+0.21s** | Sub-second cascade overhead during failover |

---

## 🔬 Benchmark Details & Artifacts

1. **Intent Detection:**
   - Raw Errors: `evaluation/results/intent_errors.csv`
   - Confusion Matrix Chart: `evaluation/results/intent_confusion_matrix.png`
   - Raw JSON: `evaluation/results/intent_results.json`

2. **Grounding Evaluation:**
   - Answers & Citations: `evaluation/results/grounding_answers.csv`
   - Accuracy Chart: `evaluation/results/grounding_accuracy.png`
   - Raw JSON: `evaluation/results/grounding_results.json`

3. **Latency Benchmarking:**
   - Comparison Chart: `evaluation/results/latency.png`
   - Raw JSON: `evaluation/results/latency_results.json`

4. **Multi-Provider Failover:**
   - Trial Log & Takeovers: `evaluation/results/failover_results.json`

---
*End of Evaluation Summary*
