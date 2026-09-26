# Failure Analysis Guide

This document explains the failure analysis process used in this lab
and provides the template and taxonomy for recording failure cases.

---

## What Is a Failure Case?

A failure case is a **specific, reproducible** instance where:
- A prompt version produces a clearly wrong, unsafe, non-compliant, or
  inconsistent response.
- The failure can be attributed to a specific prompt engineering problem.
- A fix can be designed and tested.

> A failure case is NOT a vague observation like "the response was bad".
> It is a structured record with a root cause and a proposed fix.

---

## Failure Case Template

```
Test Case ID   : <test case identifier>
Prompt Version : v1 / v2 / v3
Failure Type   : <see taxonomy below>
Description    : <exactly what went wrong — be specific>
Root Cause     : <why this happened — prompt engineering analysis>
Fix Applied    : <what was changed in the next version>
Fix Worked     : Yes / No / Partially / Unknown
Recorded At    : <ISO timestamp>
```

---

## Failure Type Taxonomy

| Type | Description |
|---|---|
| **Instruction non-compliance** | Model ignored one or more explicit instructions |
| **Format failure** | Output format did not match what was requested (JSON invalid, no bullet points, etc.) |
| **Hallucination / confabulation** | Model generated plausible-sounding but incorrect information |
| **Refusal (over-cautious)** | Model refused a legitimate, safe request |
| **Refusal (appropriate)** | Model correctly refused an unsafe request (not a bug) |
| **Safety violation** | Model produced potentially harmful content |
| **Inconsistency across runs** | Same prompt produced meaningfully different outputs in repeated runs |
| **Context ignored** | Model answered from training data instead of provided context |
| **Ambiguity exploitation** | Model made an uncontrolled assumption due to ambiguous phrasing |
| **Other** | Does not fit above categories |

---

## The Failure → Fix Cycle

```
Baseline Prompt (V1)
       │
       ▼
Observe failure (run the prompt, check output)
       │
       ▼
Classify failure (which type? why?)
       │
       ▼
Root cause analysis (which word/phrase caused the failure?)
       │
       ▼
Design V2 (targeted fix — change one thing at a time when possible)
       │
       ▼
Run V2 and compare
       │
       ▼
Did the fix work? → Record outcome
       │
       ├── Yes → Document as success
       ├── No  → Redesign → V3
       └── Partially → Consider V3 or accept trade-off
```

---

## Example Failure Cases

### Case 1: JSON Format Failure

| Field | Value |
|---|---|
| Test Case ID | `json_output` |
| Prompt Version | v1 |
| Failure Type | Format failure |
| Description | Model returned JSON wrapped in ```json markdown fences instead of raw JSON |
| Root Cause | Prompt said "as JSON" but did not specify whether markdown formatting was allowed. Model defaults to presenting code in fences. |
| Fix Applied | V2 adds: "Do not include any text before or after the JSON." V3 adds: "no markdown formatting, no code fences" and anchors output with "Start your response with '['" |
| Fix Worked | Partially (v2 helped; v3 more reliable) |

---

### Case 2: Hallucination on Unknown Entity

| Field | Value |
|---|---|
| Test Case ID | `hallucination_resistance` |
| Prompt Version | v1 |
| Failure Type | Hallucination / confabulation |
| Description | Model generated a detailed, confident historical account of "The Battle of Veloria in 1847" — a completely fictional event |
| Root Cause | No uncertainty-surfacing instruction; model defaults to generating plausible content rather than expressing uncertainty |
| Fix Applied | V2 adds explicit instruction to say if uncertain. V3 provides exact fallback phrase and bans speculation. |
| Fix Worked | V3 reliably produces the fallback (when tested) |

---

### Case 3: Context Ignored

| Field | Value |
|---|---|
| Test Case ID | `grounded_qa` |
| Prompt Version | v1 |
| Failure Type | Context ignored |
| Description | Model answered from training data about "NovaTech" (a fictional company) instead of reading the provided context |
| Root Cause | Prompt provided context but did not instruct model to use ONLY the context |
| Fix Applied | V2: "Answer using ONLY the information in the context below. Do not use outside knowledge." V3 adds explicit fallback for unanswerable questions. |
| Fix Worked | Yes — V2/V3 consistently use the provided context |

---

## Key Prompt Engineering Lessons

1. **Specificity prevents ambiguity.** Vague instructions give the model freedom to make assumptions you didn't intend.

2. **Explicit beats implicit.** If you want the model NOT to do something, say so. Silence is permission.

3. **Format constraints need anchors.** "As JSON" is weaker than "Start with '[' and end with ']', no markdown".

4. **Grounding requires explicit instruction.** "Based on the context" is weaker than "using ONLY the context; do not use outside knowledge".

5. **One change at a time.** When iterating from V1 → V2, change as little as possible so you know what caused the improvement.

6. **Improvement sometimes has trade-offs.** A stricter prompt may fix one failure mode but introduce another (over-refusal, stilted prose).

7. **Document everything.** The most valuable part of prompt engineering is the record of what you tried and why.
