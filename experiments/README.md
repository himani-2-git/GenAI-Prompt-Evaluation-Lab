# Experiments Directory

This folder stores saved experiment results as JSON files.

## What Gets Saved

Each file represents a single prompt-version run and contains:
- **test_case_id** / **test_case_name** — which test case was run
- **prompt_version** — v1, v2, or v3
- **prompt_text** — the actual prompt sent (with context filled in)
- **model** — which Gemini model was used
- **temperature** / **max_tokens** — parameter settings used
- **response_text** — the actual model output (never fabricated)
- **api_success** — whether the call succeeded
- **latency_ms** — response time
- **token usage** — input/output/total tokens (when available)
- **scores** — evaluation metrics (labelled as OBJECTIVE/HEURISTIC/GROUNDED)
- **saved_at** — UTC timestamp

## Important

> Results in this directory are **actual model outputs** from real API calls.
> No results are fabricated, pre-filled, or hardcoded.

## File Naming Convention

```
{test_case_id}_{prompt_version}_{short-uuid}.json
```

Example: `json_output_v2_3a7f9c12.json`

## Exporting

Use the Streamlit dashboard → **Experiment History** → **Export CSV** to
download all results as a spreadsheet.

You can also run:
```bash
python -c "from src.utils import load_all_experiments, experiments_to_csv; experiments_to_csv(load_all_experiments())"
```

## Note on .gitignore

Experiment JSON files are excluded from git by default (see `.gitignore`).
This is intentional:
- Results depend on your API key and model version
- Experiment files can be large
- Reproducibility is by re-running, not by committing outputs

If you want to commit specific experiments as examples, remove the pattern
from `.gitignore` or force-add specific files with `git add -f`.
