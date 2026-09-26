# Evaluation Methodology

This document explains every metric used in the GenAI Prompt Evaluation Lab,
its type, what it measures, and — critically — what it **cannot** tell you.

---

## Metric Label System

Every metric is tagged with one of four labels:

| Label | Meaning |
|---|---|
| `[OBJECTIVE]` | Directly measurable without opinion: token count, latency, API success |
| `[HEURISTIC]` | Rule-based approximation: fast and deterministic, but imperfect |
| `[GROUNDED]` | Requires a reference/ground-truth answer to be meaningful |
| `[LLM-JUDGE]` | Uses the LLM to score its own output — non-deterministic and potentially biased |

**Why this matters:** Without clear labels, a recruiter or user might mistake a heuristic score for an objective measurement. This system ensures the limitations are always visible.

---

## Objective Metrics

### `response_not_empty`
- **What it measures:** Whether the response contains at least one non-whitespace character.
- **Why it matters:** A sanity check — an empty response is always a failure.
- **Limitation:** Does not assess quality, only existence.

### `response_length_chars` / `response_length_words`
- **What it measures:** Character and word count of the response.
- **Why it matters:** Length constraints are a common prompt engineering requirement. If you ask for "one sentence" and get 500 words, that is a failure.
- **Limitation:** Word count is approximate (splits on whitespace). Does not measure whether the length is appropriate.

### `latency_ms`
- **What it measures:** Wall-clock time from sending the API request to receiving the response.
- **Why it matters:** Latency matters for user experience and is affected by model size, prompt length, and output length.
- **Limitation:** Varies with network conditions, server load, and geographic location. Not reproducible.

### `token_usage`
- **What it measures:** Input tokens (prompt), output tokens (response), and total tokens as reported by the Gemini API.
- **Why it matters:** Token usage determines API cost and is a direct measure of prompt efficiency.
- **Limitation:** Tokenization is model-specific; token counts are not directly comparable across models.

### `finish_reason`
- **What it measures:** Why the model stopped generating: `STOP` (natural end), `MAX_TOKENS` (hit limit), `SAFETY` (blocked), `RECITATION` (copyright), etc.
- **Why it matters:** `MAX_TOKENS` often means the response was truncated — a sign that `max_tokens` needs increasing.

### `api_success`
- **What it measures:** Whether the API call completed without an error.

---

## Heuristic Metrics

> ⚠️ These are approximations, not ground truth. Use them to flag potential issues, not to prove quality.

### `potential_refusal`
- **Method:** Regex pattern matching on known refusal phrases ("I'm not able to", "I can't help with").
- **Why it matters:** Detecting over-refusal is important for safety-sensitive test cases.
- **Limitations:**
  - High false-positive rate on educational content about refusals.
  - Does not detect novel or paraphrased refusals.
  - A refusal may be appropriate (safety case) or inappropriate (over-caution).

### `potential_hallucination_risk`
- **Method:** Regex matching on hedging language ("I think", "probably", "I'm not sure").
- **Why it matters:** Hedging language is a soft signal of uncertainty, which correlates with hallucination risk.
- **Limitations:**
  - Hedging is not proof of hallucination. Appropriate epistemic humility also produces hedges.
  - Confident false statements are not detected.
  - This is a surface-level heuristic.

### `safety_concern_flagged`
- **Method:** Checks if `finish_reason == "SAFETY"` or if response matches unsafe-content patterns.
- **Why it matters:** Safety compliance is a key dimension for safety-sensitive test cases.
- **Limitations:** Pattern matching may miss novel unsafe content; may false-positive on legitimate educational content.

### `instruction_keyword_adherence`
- **Method:** Fraction of key instruction terms (specified per test case) found in the response.
- **Range:** 0.0 (none found) → 1.0 (all found).
- **Why it matters:** If you asked for "three bullet points" and the response contains the words "three" and "bullet", it's a positive signal.
- **Limitations:** Keyword presence ≠ semantic compliance. A response could repeat instruction keywords without following them.

### `format_adherence`
- **Method:** Heuristic detection of the requested format (JSON parsing, bullet-point regex, markdown element detection).
- **Why it matters:** Format compliance is directly measurable for structured outputs.
- **Limitations:** JSON validation is reliable; list/markdown detection is pattern-based and imperfect.

---

## Grounded Metrics

> These metrics require a **reference answer**. Without one, they are not computed.

### `keyword_overlap_score`
- **Method:** Jaccard similarity between the word sets of the response and the reference answer.
  - `score = |response_words ∩ reference_words| / |response_words ∪ reference_words|`
  - Range: 0.0 (no word overlap) → 1.0 (identical word sets).
- **Why it matters:** A rough proxy for whether the response covers the same content as the reference.
- **Limitations:**
  - Does NOT measure semantic similarity. Synonyms score 0.
  - Stop words (the, is, a) inflate the score.
  - A paraphrase of the correct answer may score low.
  - A response that repeats reference words but answers incorrectly may score high.
  - **Not a substitute for human evaluation.**

---

## LLM-as-Judge

> ⚠️ This uses an extra API call and is non-deterministic. Treat as qualitative guidance only.

### `llm_coherence_score` / `llm_relevance_score`
- **Method:** The model is asked to score a response on a 1–5 scale and provide a one-sentence explanation.
- **Why it matters:** LLM-as-judge can catch nuanced quality issues that heuristics miss.
- **Limitations:**
  - **Self-evaluation bias:** Models tend to rate their own outputs favourably.
  - **Non-deterministic:** Different calls may return different scores.
  - **Not ground truth:** This is an opinion, not a measurement.
  - **Cost:** Each judge call consumes additional API tokens.

---

## What We Intentionally Do NOT Do

1. **No universal "accuracy" score** — There is no ground truth for open-ended generation tasks.
2. **No fabricated results** — All scores come from real computations or real API calls.
3. **No hard-coded "excellent" responses** — The system evaluates what the model actually returns.
4. **No composite "quality score"** — Combining incompatible metrics into one number is misleading.

---

## Recommended Evaluation Workflow

1. Run V1 (baseline) and record objective metrics + heuristic flags.
2. Identify which dimensions fail (format? refusal? hallucination risk?).
3. Design V2 to address those specific failures.
4. Run V2 and compare the same dimensions.
5. If grounded evaluation is possible, provide a reference answer.
6. Use LLM-judge optionally for qualitative insight.
7. Document the root cause and fix in the Failure Log.

This process is the core of systematic prompt engineering.
