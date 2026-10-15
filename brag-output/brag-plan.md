# Brag Plan: GenAI Prompt Evaluation Lab

## What is this app?
A systematic evaluation harness and Streamlit workbench for testing, measuring, and comparing LLM prompt versions with deterministic contract validation, offline reproducibility, and mathematical denominator integrity.

## The angle
Generative AI applications rarely fail from complete model outages alone—they fail from silent prompt regressions, unescaped markdown fences, token truncations, and instruction decay. The GenAI Prompt Evaluation Lab treats prompt engineering as a true software testing discipline: dual-provider abstraction, offline reproducible test mocks, 12-category failure precedence taxonomy, and 229 verified passing tests in CI.

## Hook (first 2-3 seconds)
"Stop 'Vibe Checking' Your Prompts." An impactful statement challenging ad-hoc prompt testing, followed immediately by 3 warning pills representing production failure realities: unescaped markdown fences, token truncations (`MAX_TOKENS`), and schema violations.

## Key moments (the middle)
- **Side-by-side prompt version benchmark:** Measured V1 (764 words, 4,878ms latency, 1,027 tokens, `MAX_TOKENS` truncation) vs V2 (52 words, 2,058ms latency, 89 tokens, `STOP` complete).
- **12-category deterministic failure precedence taxonomy:** Diagnostic engine strictly enforcing hierarchy from `api_error` down to `low_instruction_adherence`, distinguishing Python `bool` from `int`.
- **Mathematical denominator integrity:** Batch aggregations computed strictly over applicable test subsets (e.g. JSON schema adherence evaluated only on cases requesting JSON).

## Outro / punchline
"Treat Prompt Engineering as Software Engineering." Backed by 229 automated tests passing offline in GitHub Actions CI with zero external API credentials needed.

## User flow worth showing
1. **Configure & Benchmark:** Compare open-ended V1 prompts against constrained V2 prompts.
2. **Execute Diagnostic Evaluation:** Isolate root-cause failures in the 12-category precedence taxonomy.
3. **Batch Run & Export:** Measure delta across 12 benchmark cases and export automated CommonMark reports.

## Tone
- Preset: `polished`
- Creative direction: "Serious software engineering for generative AI systems"
- Interpretation: Restrained, elegant, authoritative dark theme with crisp metrics, legible typography, and decisive data-driven transitions.

## Format: landscape — 1920x1080
## Duration: 18.0s

## Visual identity (from the project)
- Background: `#0d1117` (Deep GitHub/VSCode dark slate)
- Card Surfaces: `#161b22` / `#21262d`
- Accent / Brand: `#38bdf8` (Cyan) and `#4285F4` (Gemini Blue)
- Success / Complete: `#34d399` (Emerald)
- Warning / Truncation: `#f87171` (Rose / Red)
- Text: `#f0f6fc` (Primary) and `#8b949e` (Muted)
- Display font: `Inter`, system-ui, -apple-system, sans-serif
- Code font: `ui-monospace`, `SFMono-Regular`, `Consolas`, monospace

## Share copy (draft)
Stop vibe-checking your prompts. The GenAI Prompt Evaluation Lab brings deterministic contract validation, 12-category failure taxonomy, and 229 passing CI tests to prompt engineering.

## Audio direction
- Role: Warm, rhythmic corporate electronic bed supporting high-precision engineering metrics.
- Music: `assets/music/bgm.mp3` (`happy-beats-business-moves-vol-1-by-ende-dot-app.mp3`)
- Music treatment: Starts at 0.0s, balanced volume bed at ~0.25, gentle fade out during outro between 17.0s and 18.0s.
- Music cue guidance: 120.19 BPM (~0.50s beat grid).
  - Beat locks: Scene transitions at 3.5s, 7.5s, 11.5s, 15.0s.
- SFX posture: Restrained, professional visual-motion sync without harsh clicks.
- Restraint rule: Audio sits cleanly under the visual story and never distracts from readability.

## Storyboard

### Scene 1 — The Hook: Stop Vibe Checking — 3.5s (0.0s - 3.5s)
- **Visual:** Dark technical grid with ambient cyan glow. Header badge: `GENAI PROMPT EVALUATION LAB`.
- **Text:** Giant bold headline: `"Stop 'Vibe Checking' Your Prompts."` Subtitle: `Silent regressions kill LLM pipelines in production.`
- **Sequential reveal:** 3 red/amber regression warning tags pop in:
  1. `⚠ Unescaped Markdown Fences`
  2. `⚠ Token Limit Truncation (MAX_TOKENS)`
  3. `⚠ Schema Contract Violations`
- **Audio intent:** Clean rhythmic groove begins.

### Scene 2 — V1 Baseline vs V2 Constrained Specification — 4.0s (3.5s - 7.5s)
- **Visual:** Side-by-side comparison cards showing real benchmark deltas from the workbench:
  - Left Card: `V1: Vague Baseline` | `764 words` | `4,878 ms` | `1,027 tokens` | `MAX_TOKENS (Truncated)`
  - Right Card: `V2: Constrained Spec` | `52 words (-93%)` | `2,058 ms (-58%)` | `89 tokens (-91%)` | `STOP (Resolved)`
- **Audio intent:** Rhythmic pulse syncs as the comparison delta highlights in bright emerald.

### Scene 3 — 12-Category Failure Precedence Taxonomy — 4.0s (7.5s - 11.5s)
- **Visual:** Engineering diagnostic inspect view.
  - Headline: `12-Category Precedence Failure Taxonomy`
  - Subhead: `Strict schema contracts · Python bool vs int differentiation`
  - Hierarchy pipeline badges: `api_error` → `empty_response` → `safety_violation` → `malformed_json` → `missing_required_keys` → `wrong_value_types`
  - Diagnostic pass card: `Criteria Passed: 3/3` | `JSON Schema: VALID` | `Hedging Risk: LOW`
- **Audio intent:** Consistent forward momentum.

### Scene 4 — Mathematical Denominator Integrity & Batch Runs — 3.5s (11.5s - 15.0s)
- **Visual:** Batch runner & reporting view.
  - Headline: `Mathematical Denominator Integrity`
  - Subhead: `12 benchmark test cases · 100% offline reproducible mock provider`
  - 3 feature pillars:
    - `Accurate Subsets (No unfair penalties)`
    - `Fair Matched Prompt Version Comparisons`
    - `Automated CommonMark Markdown Reports`
- **Audio intent:** Swell toward final resolution.

### Scene 5 — Outro & Engineering Proof — 3.0s (15.0s - 18.0s)
- **Visual:** The definitive engineering proof.
  - Core Headline: `Treat Prompt Engineering as Software Engineering`
  - 3 prominent verification badges:
    - `229 Passing Tests (Pytest)`
    - `GitHub Actions CI: Verified Passing`
    - `Streamlit Workbench (Dual-Provider)`
  - Repository link: `github.com/himani-2-git/GenAI-Prompt-Evaluation-Lab`
- **Audio intent:** Music fades smoothly to final silence at 18.0s.
