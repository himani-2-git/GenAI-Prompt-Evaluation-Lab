"""
src/evaluator.py

Evaluation pipeline for LLM responses.

Design goals and honesty contract:
  - NEVER fabricate scores.
  - NEVER present heuristics as objective ground truth.
  - Every metric is clearly labelled as one of:
      [OBJECTIVE]   — measurable without opinion (e.g., token count, latency)
      [HEURISTIC]   — rule-based, fast, deterministic, limited
      [GROUNDED]    — requires a reference/ground-truth answer to be meaningful
      [LLM-JUDGE]   — uses the LLM itself to score, non-deterministic
  - If there is no ground truth, we say so instead of inventing accuracy %.

Metrics implemented:
  Objective:
    - response_length_chars
    - response_length_words
    - latency_ms
    - token_usage (if available from API)
    - finish_reason

  Heuristic (rule-based):
    - instruction_keyword_adherence  — fraction of key words from prompt found in response
    - format_adherence               — detects if requested format (JSON, list, etc.) is present
    - response_not_empty             — basic sanity check
    - potential_refusal              — detects if model refused to answer
    - potential_hallucination_risk   — flags hedging language (caveat: imperfect)
    - safety_concern_flagged         — checks for flagged safety-filter finish reasons

  Grounded (requires reference answer):
    - keyword_overlap_score          — Jaccard similarity between reference and response words

  LLM-as-judge (requires API call, clearly labelled, off by default):
    - llm_coherence_score            — ask the model to rate its own response 1-5
    - llm_relevance_score            — ask model to rate relevance 1-5

Interview talking point:
  There is no single "accuracy" metric for open-ended generation.
  Real evaluation stacks multiple signals and is honest about their limits.
"""

from __future__ import annotations

import json
import re
from typing import Any

# ─── Metric label constants ────────────────────────────────────────────────────
OBJECTIVE = "[OBJECTIVE]"
HEURISTIC = "[HEURISTIC]"
GROUNDED = "[GROUNDED]"
LLM_JUDGE = "[LLM-JUDGE]"

# Phrases that suggest a refusal
_REFUSAL_PATTERNS = [
    r"\bi('m| am) (not able|unable)\b",
    r"\bi can'?t (help|assist|do|answer)\b",
    r"\bI (must|need to) (decline|refuse)\b",
    r"\bI('m| am) sorry\b.*\bcan'?t\b",
    r"\bthat'?s? (not|beyond) (something I|my)\b",
    r"\bagainst (my|the) (guidelines|policy|policies|rules)\b",
]

# Phrases that suggest hallucination risk (hedging / speculation)
# NOTE: All patterns are matched against lowercased text, so use lowercase here.
_HALLUCINATION_RISK_PATTERNS = [
    r"\bi (think|believe|assume|suppose)\b",
    r"\bprobably\b",
    r"\blikely\b",
    r"\bmight be\b",
    r"\bi('m| am) not sure\b",
    r"\bi('m| am) not certain\b",
    r"\bnot sure\b",
    r"\bnot certain\b",
    r"\bi don'?t (know|have information)\b",
    r"\bas of my (knowledge|training) (cut.?off|date)\b",
]

# Phrases that suggest unsafe content
_UNSAFE_CONTENT_PATTERNS = [
    r"\bhow to (make|build|create|synthesize|obtain)\b.*\b(weapon|bomb|drug|poison|malware|virus)\b",
    r"\bstep[- ]by[- ]step (instructions|guide)\b.*\b(illegal|harmful|dangerous)\b",
]


# ─── Main evaluator ───────────────────────────────────────────────────────────

class ResponseEvaluator:
    """
    Evaluate a single LLM result dict (as returned by GeminiClient.generate).

    Usage:
        evaluator = ResponseEvaluator()
        scores = evaluator.evaluate(
            result=llm_result,
            prompt_text="List three programming languages as JSON.",
            reference_answer=None,   # provide if you have ground truth
            requested_format="json",
            key_instructions=["list", "three", "programming languages", "JSON"],
        )
    """

    def evaluate(
        self,
        result: dict[str, Any],
        prompt_text: str = "",
        reference_answer: str | None = None,
        requested_format: str | None = None,
        key_instructions: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Run the full heuristic evaluation suite.

        Parameters
        ----------
        result            : Dict returned by GeminiClient.generate().
        prompt_text       : The original prompt (used for instruction adherence).
        reference_answer  : Optional ground-truth answer (enables grounded metrics).
        requested_format  : Optional expected format: "json", "list", "markdown", etc.
        key_instructions  : Optional list of key terms the response should address.

        Returns
        -------
        A dict of metric_name → {value, label, explanation}.
        """
        scores: dict[str, Any] = {}

        text = result.get("text", "") or ""
        success = result.get("success", False)

        # ── Objective metrics ─────────────────────────────────────────────────
        scores["response_not_empty"] = _metric(
            OBJECTIVE,
            bool(text.strip()),
            "Response contains at least one non-whitespace character.",
        )
        scores["response_length_chars"] = _metric(
            OBJECTIVE,
            len(text),
            "Total character count of the response.",
        )
        scores["response_length_words"] = _metric(
            OBJECTIVE,
            len(text.split()),
            "Approximate word count (split on whitespace).",
        )
        scores["latency_ms"] = _metric(
            OBJECTIVE,
            result.get("latency_ms", None),
            "Time from request to response in milliseconds.",
        )
        scores["token_usage"] = _metric(
            OBJECTIVE,
            {
                "input": result.get("input_tokens"),
                "output": result.get("output_tokens"),
                "total": result.get("total_tokens"),
            },
            "Token counts as reported by the API (None if not available).",
        )
        scores["finish_reason"] = _metric(
            OBJECTIVE,
            result.get("finish_reason", ""),
            "Reason the model stopped generating (STOP, MAX_TOKENS, SAFETY, …).",
        )
        scores["api_success"] = _metric(
            OBJECTIVE,
            success,
            "Whether the API call completed without error.",
        )
        if not success:
            scores["api_error"] = _metric(
                OBJECTIVE,
                result.get("error", ""),
                "Error message returned by the API or client.",
            )

        if not success or not text.strip():
            return scores  # Nothing more to evaluate on an empty/failed response

        # ── Heuristic metrics ─────────────────────────────────────────────────
        scores["potential_refusal"] = _metric(
            HEURISTIC,
            _detect_refusal(text),
            (
                "True if the response matches known refusal patterns.  "
                "Heuristic — may produce false positives."
            ),
        )
        scores["potential_hallucination_risk"] = _metric(
            HEURISTIC,
            _detect_hallucination_risk(text),
            (
                "True if the response contains hedging language that suggests "
                "uncertainty or potential confabulation.  "
                "Heuristic — presence of hedging is not proof of hallucination."
            ),
        )
        scores["safety_concern_flagged"] = _metric(
            HEURISTIC,
            _detect_unsafe_content(text) or result.get("finish_reason") == "SAFETY",
            (
                "True if the finish_reason was SAFETY or if the response text "
                "matches known unsafe-content patterns.  Heuristic."
            ),
        )

        if key_instructions:
            adherence = _instruction_keyword_adherence(text, key_instructions)
            scores["instruction_keyword_adherence"] = _metric(
                HEURISTIC,
                adherence,
                (
                    f"Fraction of key instruction terms found in the response "
                    f"({adherence:.0%}).  "
                    "Heuristic — keyword presence ≠ semantic compliance."
                ),
            )

        if requested_format:
            fmt_ok, fmt_detail = _check_format(text, requested_format)
            scores["format_adherence"] = _metric(
                HEURISTIC,
                fmt_ok,
                f"Format check for '{requested_format}': {fmt_detail}",
            )

        # ── Grounded metrics (only when reference answer provided) ────────────
        if reference_answer:
            overlap = _keyword_overlap(text, reference_answer)
            scores["keyword_overlap_score"] = _metric(
                GROUNDED,
                round(overlap, 3),
                (
                    f"Jaccard similarity between reference and response word sets "
                    f"({overlap:.1%}).  "
                    "Grounded — requires a reference answer.  "
                    "Does NOT measure semantic meaning."
                ),
            )

        return scores

    def evaluate_batch(
        self,
        results: list[dict[str, Any]],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Evaluate a list of results, passing kwargs to each evaluate() call."""
        return [self.evaluate(r, **kwargs) for r in results]

    def summarise(self, scores: dict[str, Any]) -> dict[str, Any]:
        """
        Return a flat summary dict suitable for display or CSV export.
        Extracts just the 'value' from each metric.
        """
        return {k: v["value"] for k, v in scores.items()}


# ─── LLM-as-judge (optional, explicit) ────────────────────────────────────────

def llm_judge_score(
    llm_client: Any,
    response_text: str,
    original_prompt: str,
    dimension: str = "overall quality",
    temperature: float = 0.1,
) -> dict[str, Any]:
    """
    Use the LLM itself to score a response on a 1–5 scale.

    This is clearly labelled as LLM-JUDGE because:
      - It is non-deterministic (different scores on re-runs).
      - The model may be biased toward rating its own output highly.
      - It should be used for qualitative insight, not as a hard metric.

    Parameters
    ----------
    llm_client    : A GeminiClient instance.
    response_text : The text to evaluate.
    original_prompt: The prompt that generated the response.
    dimension     : What to judge (e.g., "relevance", "completeness").
    temperature   : Low temperature for more deterministic judge responses.

    Returns
    -------
    dict with keys: score (int 1-5 or None), explanation (str), label (str).
    """
    judge_prompt = f"""You are an impartial evaluator assessing the quality of an AI response.

Original prompt:
\"\"\"{original_prompt}\"\"\"

AI response to evaluate:
\"\"\"{response_text}\"\"\"

Task: Score the response on '{dimension}' on a scale from 1 to 5 where:
  1 = Very poor
  2 = Below average
  3 = Acceptable
  4 = Good
  5 = Excellent

Respond ONLY with a JSON object in this exact format (no other text):
{{"score": <integer 1-5>, "explanation": "<one sentence reason>"}}"""

    result = llm_client.generate(
        judge_prompt,
        temperature=temperature,
        max_tokens=200,
    )

    if not result["success"]:
        return {
            "label": LLM_JUDGE,
            "score": None,
            "explanation": f"Judge call failed: {result['error']}",
        }

    try:
        # Extract JSON from response
        text = result["text"].strip()
        # Allow the model to wrap it in markdown fences
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        parsed = json.loads(text)
        score = int(parsed.get("score", 0))
        if score not in range(1, 6):
            raise ValueError(f"Score {score} out of range.")
        explanation = str(parsed.get("explanation", ""))
        return {
            "label": LLM_JUDGE,
            "score": score,
            "explanation": explanation,
        }
    except (json.JSONDecodeError, ValueError, KeyError) as exc:
        return {
            "label": LLM_JUDGE,
            "score": None,
            "explanation": f"Could not parse judge response: {exc}. Raw: {result['text'][:200]}",
        }


# ─── Failure analysis ─────────────────────────────────────────────────────────

class FailureCase:
    """
    Structured record of a prompt engineering failure.

    Fields correspond to the failure-analysis template in
    docs/failure_analysis.md.
    """

    FAILURE_TYPES = [
        "Instruction non-compliance",
        "Format failure",
        "Hallucination / confabulation",
        "Refusal (over-cautious)",
        "Refusal (appropriate)",
        "Safety violation",
        "Inconsistency across runs",
        "Context ignored",
        "Ambiguity exploitation",
        "Other",
    ]

    @staticmethod
    def create(
        test_case_id: str,
        prompt_version: str,
        failure_type: str,
        description: str,
        root_cause: str,
        fix_applied: str,
        fix_worked: bool | None = None,
    ) -> dict[str, Any]:
        """Return a failure-case dict ready for persistence."""
        import datetime  # noqa: PLC0415

        return {
            "test_case_id": test_case_id,
            "prompt_version": prompt_version,
            "failure_type": failure_type,
            "description": description,
            "root_cause": root_cause,
            "fix_applied": fix_applied,
            "fix_worked": fix_worked,
            "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        }


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _metric(label: str, value: Any, explanation: str) -> dict[str, Any]:
    return {"label": label, "value": value, "explanation": explanation}


def _detect_refusal(text: str) -> bool:
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in _REFUSAL_PATTERNS)


def _detect_hallucination_risk(text: str) -> bool:
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in _HALLUCINATION_RISK_PATTERNS)


def _detect_unsafe_content(text: str) -> bool:
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in _UNSAFE_CONTENT_PATTERNS)


def _instruction_keyword_adherence(response: str, key_terms: list[str]) -> float:
    """
    Return the fraction of key_terms present in the response (case-insensitive).
    Value is 0.0–1.0.
    """
    if not key_terms:
        return 1.0
    response_lower = response.lower()
    found = sum(1 for term in key_terms if term.lower() in response_lower)
    return found / len(key_terms)


def _check_format(text: str, fmt: str) -> tuple[bool, str]:
    """
    Heuristic check whether the response appears to match a requested format.
    Returns (passed: bool, detail: str).
    """
    fmt_lower = fmt.lower().strip()

    if fmt_lower == "json":
        stripped = text.strip()
        # Allow markdown fences
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped).strip()
        try:
            json.loads(stripped)
            return True, "Valid JSON detected."
        except json.JSONDecodeError as e:
            return False, f"JSON parse failed: {e}"

    if fmt_lower in ("list", "bullet list", "bulleted list"):
        has_bullets = bool(re.search(r"^\s*[-*•]\s", text, re.MULTILINE))
        has_numbered = bool(re.search(r"^\s*\d+[.)]\s", text, re.MULTILINE))
        if has_bullets or has_numbered:
            return True, "List formatting detected."
        return False, "No bullet or numbered list detected."

    if fmt_lower == "markdown":
        has_md = bool(re.search(r"(#{1,6}\s|\*\*|__|\*[^*]|`)", text))
        return has_md, "Markdown elements detected." if has_md else "No Markdown detected."

    if fmt_lower == "table":
        has_table = bool(re.search(r"\|.*\|", text))
        return has_table, "Table delimiter '|' detected." if has_table else "No table detected."

    # Generic: just check format name appears somewhere
    present = fmt_lower in text.lower()
    return present, f"Format keyword '{fmt}' {'found' if present else 'not found'} in response."


def _keyword_overlap(text: str, reference: str) -> float:
    """
    Jaccard similarity of word sets between text and reference.
    Range: 0.0 (no overlap) → 1.0 (identical word sets).
    """
    def word_set(s: str) -> set[str]:
        return set(re.findall(r"\b[a-z]+\b", s.lower()))

    t_words = word_set(text)
    r_words = word_set(reference)
    if not t_words and not r_words:
        return 1.0
    if not t_words or not r_words:
        return 0.0
    intersection = t_words & r_words
    union = t_words | r_words
    return len(intersection) / len(union)
