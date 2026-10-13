# 🚀 Beginner's Quickstart Guide — Commands Cheat Sheet

Welcome! If you are new to Python, virtual environments, or Streamlit, this guide walks you through every step with the exact commands to copy and paste.

---

## 📌 All Commands at a Glance

| Action | Command | What It Does |
|---|---|---|
| **1. Activate Environment** | `.\venv\Scripts\Activate.ps1` | Activates your Python environment `(venv)` |
| **2. Test API Connection** | `python tests\test_api_connection.py` | Checks if your Gemini API key is working |
| **3. Launch Web App** | `python -m streamlit run app.py` | Starts the interactive web dashboard |
| **4. Open App in Browser** | Open **[http://localhost:8501](http://localhost:8501)** | View and play with the app |
| **5. Run All Tests** | `pytest` | Runs all 229 automated unit and regression tests |
| **6. Stop the Web App** | Press `Ctrl + C` in the terminal | Shuts down the local web server |
| **7. Exit Environment** | `deactivate` | Exits the `(venv)` environment |

---

## 📝 Step-by-Step Walkthrough

### Step 1: Open Your Terminal & Activate the Virtual Environment

1. Open PowerShell in your project folder (`c:\Users\enhim\projects\GenAI-Prompt-Evaluation-Lab`).
2. Run this command:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
3. Look at your prompt. You should see `(venv)` at the beginning:
   ```text
   (venv) PS C:\Users\enhim\projects\GenAI-Prompt-Evaluation-Lab>
   ```

> [!TIP]
> **Got a red error saying "script execution is disabled"?**  
> Run this command once in PowerShell:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
> ```
> Then run `.\venv\Scripts\Activate.ps1` again.

---

### Step 2: Verify Your Gemini API Key (Optional for Offline Mock)

If you plan to use Google Gemini live models, make sure your Gemini API key in `.env` is working properly:

```powershell
python tests\test_api_connection.py
```

**Expected result:**
```text
Key loaded: True
Model: gemini-flash-latest
Sending test prompt...

Success:     True
Response:    'API connection successful.'
```
If you see `Success: True`, you are ready to make live calls!

> [!NOTE]
> If you do not have an API key yet, you can skip this step. The app includes a **Deterministic Mock** provider that runs 100% offline without any API key or network connection.

---

### Step 3: Run the Web Application

Launch the Streamlit interface:

```powershell
python -m streamlit run app.py
```

> [!NOTE]
> Why `python -m streamlit` instead of just `streamlit`?  
> On Windows, running `python -m streamlit` ensures Windows Application Control policies never block the command.

---

### Step 4: Open and Explore the Dashboard

Open your web browser (Chrome, Edge, etc.) and go to:
👉 **[http://localhost:8501](http://localhost:8501)**

Inside the app, you will find 7 sections on the left sidebar:
1. **🚀 Run Experiment** — Choose from 12 benchmark test cases (like JSON formatting or summarization) or custom inputs, select Gemini or Deterministic Mock, and click **Run All Versions** to see generated responses for V1, V2, and V3 prompts.
2. **✏️ Custom Prompt** — Write and test ad-hoc prompt variations against any test case.
3. **📊 Compare Outputs** — View different prompt outputs side-by-side with latency and token usage.
4. **🔍 Evaluate** — Run automated quality checks (JSON validation, word limits, refusal detection, grounded overlap).
5. **📦 Batch Evaluation** — Run all prompt versions across benchmark test cases, view aggregate metrics, and download Markdown/CSV reports.
6. **⚠️ Failure Log** — Browse and document failure modes categorized by a 12-category taxonomy.
7. **📋 Experiment History** — View all past runs or download them as a CSV.

---

### Step 5: Run Automated Tests

To ensure all code and evaluation components are functioning without using your API quota:

```powershell
pytest
```

**Expected output:**
```text
tests\test_api_connection.py .
tests\test_app_workbench.py .....................
tests\test_batch_runner.py ................................
tests\test_evaluator.py ................................................
tests\test_integration.py ....
tests\test_llm_client.py ................................
tests\test_prompt_manager.py .......................................
tests\test_utils.py ......................................
============================= 229 passed in 5.01s =============================
```

To see each test name as it passes, add `-v`:
```powershell
pytest -v
```

---

### Step 6: When You Are Done

1. In the terminal where the web app is running, press **`Ctrl + C`** on your keyboard to stop the server.
2. To exit the virtual environment, type:
   ```powershell
   deactivate
   ```

---

## ❓ Common Questions & Troubleshooting

#### "Port 8501 is already in use"
If Streamlit says port 8501 is already occupied, it will automatically switch to `http://localhost:8502`. Check the URL printed in your terminal.

#### "GEMINI_API_KEY is not set" or "API key contains placeholder"
Make sure you have a file named `.env` in the root folder with:
```env
GEMINI_API_KEY="your-actual-api-key-here"
```
Get a free API key at [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey).

> [!TIP]
> **No API key?** You can switch the LLM Provider in the sidebar to **Deterministic Mock (offline)** to run benchmark experiments, comparisons, evaluations, and batch reports completely offline!
