"""
app.py — GenAI Prompt Evaluation Lab

Entry point for the Streamlit dashboard.

The app is organized into five pages/sections rendered in the sidebar:
  1. Run Experiment  — pick test case, set params, run 1-3 prompt versions
  2. Compare         — side-by-side output comparison
  3. Evaluate        — heuristic + LLM-judge scores
  4. Failure Log     — record and browse failure cases
  5. History         — browse all past runs, export to CSV

Run with:
    streamlit run app.py

Interview talking point:
  Streamlit turns Python functions into interactive UIs without any
  JavaScript.  Each re-run of the script IS the app's render cycle —
  there is no separate server loop.
"""

import json
import sys
from pathlib import Path

import streamlit as st

# ─── Make src importable ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.evaluator import FailureCase, ResponseEvaluator, llm_judge_score
from src.llm_client import GeminiClient, SUPPORTED_MODELS
from src.utils import (
    build_experiment_record,
    experiments_to_csv,
    friendly_timestamp,
    load_all_experiments,
    load_failure_cases,
    save_experiment,
    save_failure_case,
    truncate,
    utc_now,
)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GenAI Prompt Evaluation Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Load test cases ──────────────────────────────────────────────────────────
@st.cache_data
def load_test_cases() -> list[dict]:
    tc_path = Path(__file__).parent / "test_cases" / "sample_cases.json"
    with tc_path.open(encoding="utf-8") as f:
        return json.load(f)


TEST_CASES = load_test_cases()
TC_BY_ID = {tc["id"]: tc for tc in TEST_CASES}


# ─── Sidebar navigation ───────────────────────────────────────────────────────
st.sidebar.title("🧪 Prompt Eval Lab")
st.sidebar.markdown("*Prompt Design · Testing · Evaluation*")
st.sidebar.divider()

PAGE = st.sidebar.radio(
    "Navigate",
    [
        "🚀 Run Experiment",
        "📊 Compare Outputs",
        "🔍 Evaluate",
        "⚠️ Failure Log",
        "📋 Experiment History",
    ],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.markdown(
    "**Tip:** Run an experiment first, then use Compare and Evaluate to analyse results."
)


# ─── API key status ───────────────────────────────────────────────────────────
from dotenv import load_dotenv  # noqa: E402
load_dotenv()
import os  # noqa: E402

api_key = os.getenv("GEMINI_API_KEY", "")
if api_key and api_key != "your_gemini_api_key_here":
    st.sidebar.success("✅ API key loaded")
else:
    st.sidebar.error("❌ No API key — add GEMINI_API_KEY to .env")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1: Run Experiment
# ═════════════════════════════════════════════════════════════════════════════
if PAGE == "🚀 Run Experiment":
    st.title("🚀 Run Experiment")
    st.markdown(
        "Select a test case, configure model parameters, choose which prompt "
        "versions to run, and execute them against the Gemini API."
    )

    col_left, col_right = st.columns([2, 1])

    with col_left:
        # ── Test case selection ────────────────────────────────────────────
        tc_names = [f"{tc['name']}" for tc in TEST_CASES]
        tc_idx = st.selectbox("Test Case", range(len(tc_names)), format_func=lambda i: tc_names[i])
        selected_tc = TEST_CASES[tc_idx]

        with st.expander("ℹ️ About this test case", expanded=True):
            st.markdown(f"**Category:** `{selected_tc['category']}`")
            st.markdown(f"**Problem:** {selected_tc['problem_statement']}")
            if selected_tc.get("context"):
                st.markdown("**Context provided to model:**")
                st.code(selected_tc["context"][:400] + ("…" if len(selected_tc["context"]) > 400 else ""), language=None)

    with col_right:
        # ── Model parameters ──────────────────────────────────────────────
        st.markdown("### ⚙️ Model Parameters")

        model_name = st.selectbox(
            "Model",
            list(SUPPORTED_MODELS.keys()),
            format_func=lambda k: SUPPORTED_MODELS[k],
            help="Free-tier models: gemini-1.5-flash is fastest and most generous.",
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.05,
            help=(
                "Controls randomness. 0.0 = deterministic (same output each run). "
                "Higher = more creative and varied. For evaluation, lower is better."
            ),
        )
        st.caption(f"{'🧊 Deterministic' if temperature < 0.3 else '🌡️ Balanced' if temperature < 1.2 else '🔥 Creative'}")

        max_tokens = st.slider(
            "Max Output Tokens",
            min_value=64,
            max_value=2048,
            value=512,
            step=64,
            help="Upper bound on response length. 1 token ≈ 0.75 words.",
        )

        system_instruction = st.text_area(
            "System Instruction (optional)",
            value="",
            height=80,
            help="A persistent instruction prepended before the user prompt (supported by 1.5 models).",
        )

    st.divider()

    # ── Prompt version selector ────────────────────────────────────────────
    st.markdown("### 📝 Prompt Versions to Run")
    versions_available = list(selected_tc["prompts"].keys())
    versions_to_run = st.multiselect(
        "Select versions",
        versions_available,
        default=versions_available[:2],
        help="Run multiple versions to compare them side-by-side.",
    )

    # Show prompt previews
    if versions_to_run:
        tabs = st.tabs([f"Prompt {v.upper()}" for v in versions_to_run])
        for tab, v in zip(tabs, versions_to_run):
            with tab:
                pdata = selected_tc["prompts"][v]
                st.markdown(f"**Rationale:** {pdata['rationale']}")
                st.markdown(f"**Expected failure mode (v1):** {pdata['failure_mode']}")
                st.code(pdata["text"], language=None)

    st.divider()

    # ── Run button ────────────────────────────────────────────────────────
    if not versions_to_run:
        st.warning("Select at least one prompt version to run.")
    elif not api_key or api_key == "your_gemini_api_key_here":
        st.error("Add your GEMINI_API_KEY to the .env file to run experiments.")
    else:
        if st.button("▶️ Run Selected Prompts", type="primary", use_container_width=True):
            client = GeminiClient(api_key=api_key, model=model_name)
            evaluator = ResponseEvaluator()
            model_params = {"temperature": temperature, "max_tokens": max_tokens}

            st.session_state["last_results"] = []
            st.session_state["last_tc"] = selected_tc
            st.session_state["last_model_params"] = model_params

            for v in versions_to_run:
                pdata = selected_tc["prompts"][v]
                prompt_text = pdata["text"]

                # Fill in context/user_question placeholders
                ctx = selected_tc.get("context") or ""
                uq = (selected_tc.get("user_question_variants") or [""])[0]
                prompt_filled = prompt_text.replace("{context}", ctx).replace("{user_question}", uq)

                with st.spinner(f"Running {v.upper()}…"):
                    result = client.generate(
                        prompt=prompt_filled,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        system_instruction=system_instruction,
                    )
                    result["prompt_version"] = v

                    # Evaluate
                    scores = evaluator.evaluate(
                        result=result,
                        prompt_text=prompt_filled,
                        reference_answer=selected_tc.get("reference_answer"),
                        requested_format=selected_tc.get(f"requested_format_{v}"),
                        key_instructions=selected_tc.get(f"key_instructions_{v}"),
                    )

                    # Build and save record
                    record = build_experiment_record(
                        test_case=selected_tc,
                        prompt_version=v,
                        prompt_text=prompt_filled,
                        llm_result=result,
                        scores=scores,
                        model_params=model_params,
                    )
                    saved_path = save_experiment(record)
                    result["_record"] = record
                    result["_scores"] = scores
                    result["_saved_path"] = str(saved_path)
                    st.session_state["last_results"].append(result)

            st.success(f"✅ Ran {len(versions_to_run)} prompt version(s). Results saved.")
            st.info("👉 Go to **Compare Outputs** or **Evaluate** to analyse results.")

    # ── Show last results inline ───────────────────────────────────────────
    if "last_results" in st.session_state and st.session_state["last_results"]:
        st.divider()
        st.markdown("### 📄 Last Run Outputs")
        for r in st.session_state["last_results"]:
            v = r["prompt_version"]
            with st.expander(f"Prompt {v.upper()} — {'✅ Success' if r['success'] else '❌ Failed'}", expanded=True):
                if r["success"]:
                    st.markdown(r["text"])
                    st.caption(
                        f"Latency: {r.get('latency_ms', '?'):.0f}ms | "
                        f"Tokens: {r.get('total_tokens', 'N/A')} | "
                        f"Finish: {r.get('finish_reason', '?')}"
                    )
                else:
                    st.error(f"Error: {r['error']}")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2: Compare Outputs
# ═════════════════════════════════════════════════════════════════════════════
elif PAGE == "📊 Compare Outputs":
    st.title("📊 Side-by-Side Comparison")
    st.markdown(
        "Compare outputs from multiple prompt versions to see how each modification "
        "changed the model's behaviour."
    )

    results = st.session_state.get("last_results", [])
    if not results:
        st.info("No results in session yet. Run an experiment first.")
        # Allow loading from history
        all_exps = load_all_experiments()
        if all_exps:
            st.markdown("### Or load from history")
            tc_ids = sorted({e.get("test_case_id", "") for e in all_exps})
            selected_tid = st.selectbox("Filter by test case", ["All"] + tc_ids)
            filtered = [e for e in all_exps if selected_tid == "All" or e.get("test_case_id") == selected_tid]
            if filtered:
                # Group by test_case_id and show comparison
                st.markdown(f"Showing {len(filtered)} saved result(s).")
                versions_available = sorted({e.get("prompt_version", "") for e in filtered})
                v_select = st.multiselect("Compare versions", versions_available, default=versions_available[:2])
                show = [e for e in filtered if e.get("prompt_version") in v_select]
                if show:
                    cols = st.columns(len(show))
                    for col, exp in zip(cols, show):
                        with col:
                            v = exp.get("prompt_version", "?")
                            st.markdown(f"#### Prompt {v.upper()}")
                            st.caption(
                                f"Model: `{exp.get('model', '?')}` | "
                                f"Temp: `{exp.get('temperature', '?')}` | "
                                f"{friendly_timestamp(exp.get('saved_at', ''))}"
                            )
                            if exp.get("api_success"):
                                st.markdown(exp.get("response_text", ""))
                            else:
                                st.error(exp.get("api_error", "Unknown error"))
        st.stop()

    tc = st.session_state.get("last_tc", {})
    params = st.session_state.get("last_model_params", {})

    st.markdown(f"**Test Case:** {tc.get('name', '')} | **Model params:** temp={params.get('temperature')} max_tokens={params.get('max_tokens')}")
    st.divider()

    cols = st.columns(len(results))
    for col, r in zip(cols, results):
        v = r["prompt_version"]
        with col:
            st.markdown(f"#### Prompt {v.upper()}")
            pdata = tc.get("prompts", {}).get(v, {})
            with st.expander("Prompt text", expanded=False):
                st.code(r.get("prompt", ""), language=None)
            st.caption(f"Rationale: *{pdata.get('rationale', 'N/A')}*")
            st.divider()
            if r["success"]:
                st.markdown(r["text"])
                st.caption(
                    f"⏱ {r.get('latency_ms', '?'):.0f}ms | "
                    f"🔤 {len(r.get('text', '').split())} words | "
                    f"🎫 {r.get('total_tokens', 'N/A')} tokens"
                )
            else:
                st.error(f"❌ {r['error']}")

    # ── Metrics table ──────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📈 Quick Metrics")
    import pandas as pd  # noqa: PLC0415
    rows = []
    for r in results:
        rows.append({
            "Version": r["prompt_version"].upper(),
            "Success": "✅" if r["success"] else "❌",
            "Words": len(r.get("text", "").split()) if r["success"] else 0,
            "Latency (ms)": f"{r.get('latency_ms', 0):.0f}",
            "Tokens": r.get("total_tokens", "N/A"),
            "Finish Reason": r.get("finish_reason", "—"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3: Evaluate
# ═════════════════════════════════════════════════════════════════════════════
elif PAGE == "🔍 Evaluate":
    st.title("🔍 Evaluation")
    st.markdown(
        "View heuristic evaluation scores for the last experiment run.  \n"
        "All scores are labelled with their method: "
        "`[OBJECTIVE]` `[HEURISTIC]` `[GROUNDED]` `[LLM-JUDGE]`"
    )

    results = st.session_state.get("last_results", [])
    if not results:
        st.info("Run an experiment first to see evaluation scores.")
        st.stop()

    tc = st.session_state.get("last_tc", {})

    for r in results:
        v = r["prompt_version"]
        scores = r.get("_scores", {})

        st.markdown(f"### Prompt {v.upper()}")
        if not r["success"]:
            st.error(f"Run failed: {r['error']}")
            continue

        # ── Objective metrics ──────────────────────────────────────────────
        st.markdown("#### 📏 Objective Metrics")
        obj_cols = st.columns(4)
        obj_cols[0].metric("Length (words)", scores.get("response_length_words", {}).get("value", "—"))
        obj_cols[1].metric("Latency (ms)", f"{scores.get('latency_ms', {}).get('value', 0):.0f}")
        obj_cols[2].metric("Total Tokens", scores.get("token_usage", {}).get("value", {}).get("total", "N/A"))
        obj_cols[3].metric("Finish Reason", scores.get("finish_reason", {}).get("value", "—") or "—")

        # ── Heuristic metrics ──────────────────────────────────────────────
        st.markdown("#### 🔎 Heuristic Metrics *(rule-based — not objective truth)*")
        heuristic_keys = [
            "potential_refusal",
            "potential_hallucination_risk",
            "safety_concern_flagged",
            "instruction_keyword_adherence",
            "format_adherence",
        ]
        for k in heuristic_keys:
            if k in scores:
                m = scores[k]
                val = m["value"]
                label = m["label"]
                explanation = m["explanation"]
                if isinstance(val, bool):
                    icon = "⚠️" if val else "✅"
                    st.markdown(f"**{icon} {k.replace('_', ' ').title()}** `{label}`")
                elif isinstance(val, float):
                    pct = f"{val:.0%}"
                    icon = "✅" if val >= 0.7 else "⚠️" if val >= 0.4 else "❌"
                    st.markdown(f"**{icon} {k.replace('_', ' ').title()}** `{label}` → `{pct}`")
                else:
                    st.markdown(f"**{k.replace('_', ' ').title()}** `{label}` → `{val}`")
                st.caption(explanation)

        # ── Grounded metrics ───────────────────────────────────────────────
        if "keyword_overlap_score" in scores:
            m = scores["keyword_overlap_score"]
            st.markdown("#### 📐 Grounded Metrics *(reference answer required)*")
            st.markdown(
                f"**Keyword Overlap Score** `{m['label']}` → `{m['value']:.3f}` "
                f"({m['value']:.1%})"
            )
            st.caption(m["explanation"])

        # ── LLM-as-judge ──────────────────────────────────────────────────
        st.markdown("#### 🤖 LLM-as-Judge *(optional, non-deterministic)*")
        st.info(
            "LLM-judge scoring uses an extra API call to ask the model to score "
            "its own response. This is approximate and non-deterministic — treat as "
            "qualitative guidance only."
        )
        if api_key and api_key != "your_gemini_api_key_here":
            if st.button(f"Run LLM Judge for {v.upper()}", key=f"judge_{v}"):
                client = GeminiClient(api_key=api_key)
                with st.spinner("Asking model to judge its response…"):
                    judge = llm_judge_score(
                        llm_client=client,
                        response_text=r["text"],
                        original_prompt=r.get("prompt", ""),
                        dimension="overall relevance and instruction adherence",
                    )
                if judge["score"] is not None:
                    st.metric("Judge Score (1–5)", f"{judge['score']}/5")
                    st.caption(f"Explanation: {judge['explanation']}")
                    st.caption("Label: `[LLM-JUDGE]` — non-deterministic, qualitative only.")
                else:
                    st.warning(f"Judge failed: {judge['explanation']}")
        else:
            st.caption("Add API key to enable LLM-judge.")

        st.divider()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4: Failure Log
# ═════════════════════════════════════════════════════════════════════════════
elif PAGE == "⚠️ Failure Log":
    st.title("⚠️ Failure Log")
    st.markdown(
        "Record and browse documented failure cases.  \n"
        "A *failure case* is a specific, reproducible prompt-engineering problem "
        "with a documented root cause and fix."
    )

    # ── Log a new failure ──────────────────────────────────────────────────
    with st.expander("➕ Log a New Failure Case", expanded=False):
        with st.form("failure_form"):
            tc_ids = [tc["id"] for tc in TEST_CASES]
            tc_names_map = {tc["id"]: tc["name"] for tc in TEST_CASES}
            fail_tc = st.selectbox("Test Case", tc_ids, format_func=lambda x: tc_names_map[x])
            fail_v = st.selectbox("Prompt Version", ["v1", "v2", "v3"])
            fail_type = st.selectbox("Failure Type", FailureCase.FAILURE_TYPES)
            fail_desc = st.text_area("Description of what went wrong", height=80)
            fail_cause = st.text_area("Root cause analysis", height=80)
            fail_fix = st.text_area("Fix applied (or proposed fix)", height=80)
            fail_worked = st.selectbox("Did the fix work?", ["Unknown", "Yes", "No", "Partially"])

            submitted = st.form_submit_button("Save Failure Case")
            if submitted:
                worked_map = {"Unknown": None, "Yes": True, "No": False, "Partially": "partial"}
                failure = FailureCase.create(
                    test_case_id=fail_tc,
                    prompt_version=fail_v,
                    failure_type=fail_type,
                    description=fail_desc,
                    root_cause=fail_cause,
                    fix_applied=fail_fix,
                    fix_worked=worked_map[fail_worked],
                )
                save_failure_case(failure)
                st.success("Failure case saved.")

    # ── Browse existing failures ───────────────────────────────────────────
    st.markdown("### Logged Failure Cases")
    failures = load_failure_cases()
    if not failures:
        st.info("No failure cases logged yet.  Use the form above to record your first one.")
    else:
        import pandas as pd  # noqa: PLC0415
        df = pd.DataFrame(failures)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Detail view
        if len(failures) > 0:
            sel_idx = st.selectbox("View detail", range(len(failures)), format_func=lambda i: f"#{i+1} — {failures[i]['test_case_id']} {failures[i]['prompt_version']} — {failures[i]['failure_type']}")
            fc = failures[sel_idx]
            st.markdown(f"**Test Case:** `{fc['test_case_id']}` | **Version:** `{fc['prompt_version']}`")
            st.markdown(f"**Failure Type:** {fc['failure_type']}")
            st.markdown(f"**Description:** {fc['description']}")
            st.markdown(f"**Root Cause:** {fc['root_cause']}")
            st.markdown(f"**Fix Applied:** {fc['fix_applied']}")
            st.markdown(f"**Fix Worked:** {fc.get('fix_worked', 'Unknown')}")
            st.caption(f"Recorded at: {friendly_timestamp(fc.get('recorded_at', ''))}")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5: Experiment History
# ═════════════════════════════════════════════════════════════════════════════
elif PAGE == "📋 Experiment History":
    st.title("📋 Experiment History")
    st.markdown("Browse all saved experiment results. Results are the actual model outputs — not fabricated.")

    all_exps = load_all_experiments()
    if not all_exps:
        st.info("No experiments saved yet. Run an experiment to populate history.")
        st.stop()

    import pandas as pd  # noqa: PLC0415

    # ── Filters ────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        tc_filter = st.multiselect(
            "Test Case",
            sorted({e.get("test_case_id", "") for e in all_exps}),
            default=[],
        )
    with col2:
        v_filter = st.multiselect(
            "Prompt Version",
            sorted({e.get("prompt_version", "") for e in all_exps}),
            default=[],
        )
    with col3:
        model_filter = st.multiselect(
            "Model",
            sorted({e.get("model", "") for e in all_exps}),
            default=[],
        )

    filtered = all_exps
    if tc_filter:
        filtered = [e for e in filtered if e.get("test_case_id") in tc_filter]
    if v_filter:
        filtered = [e for e in filtered if e.get("prompt_version") in v_filter]
    if model_filter:
        filtered = [e for e in filtered if e.get("model") in model_filter]

    st.markdown(f"**Showing {len(filtered)} of {len(all_exps)} results**")

    # ── Summary table ──────────────────────────────────────────────────────
    table_rows = []
    for e in filtered:
        table_rows.append({
            "ID": e.get("id", ""),
            "Test Case": e.get("test_case_name", e.get("test_case_id", "")),
            "Version": e.get("prompt_version", ""),
            "Model": e.get("model", ""),
            "Temp": e.get("temperature", ""),
            "Success": "✅" if e.get("api_success") else "❌",
            "Words": len(e.get("response_text", "").split()),
            "Latency (ms)": f"{e.get('latency_ms', 0):.0f}",
            "Tokens": e.get("total_tokens", "N/A"),
            "Saved": friendly_timestamp(e.get("saved_at", "")),
        })

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # ── Detail view ────────────────────────────────────────────────────────
    if filtered:
        sel = st.selectbox(
            "Inspect a result",
            range(len(filtered)),
            format_func=lambda i: (
                f"#{i+1} — {filtered[i].get('test_case_id', '')} "
                f"{filtered[i].get('prompt_version', '')} — "
                f"{friendly_timestamp(filtered[i].get('saved_at', ''))}"
            ),
        )
        exp = filtered[sel]
        with st.expander("Full Result", expanded=True):
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.markdown("**Response:**")
                if exp.get("api_success"):
                    st.markdown(exp.get("response_text", ""))
                else:
                    st.error(exp.get("api_error", "Unknown error"))
            with col_b:
                st.markdown("**Evaluation Scores:**")
                scores = exp.get("scores", {})
                for k, m in scores.items():
                    if isinstance(m, dict) and "value" in m:
                        val = m["value"]
                        if isinstance(val, (bool, float, int, str)) and val != {} and val is not None:
                            st.caption(f"`{m.get('label', '')}` **{k}**: {val}")

    # ── Export ─────────────────────────────────────────────────────────────
    st.divider()
    if st.button("📥 Export All Results to CSV"):
        csv_path = experiments_to_csv(all_exps)
        with open(csv_path, encoding="utf-8") as f:
            csv_data = f.read()
        st.download_button(
            "⬇️ Download CSV",
            data=csv_data,
            file_name="prompt_experiments.csv",
            mime="text/csv",
        )
        st.success(f"CSV ready: {csv_path}")
