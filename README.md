# OpenEnv: Advanced Multi-Table Data Engineering Pipeline

tags: [openenv]

A production-grade AI evaluation environment that tests agentic capabilities on realistic, multi-table data engineering tasks. Unlike toy environments, this pipeline features schema drift, join corruption, nested JSON corruption, and adversarial log signals — mirroring real-world data lake challenges.

## Why This Matters

In production, data engineers spend 60-80% of their time on data cleaning and reconciliation. This environment tests whether AI agents can handle the messy reality of enterprise data: inconsistent schemas across tables, silent join failures, corrupted nested structures, and misleading diagnostic logs.

## The Challenge

Three progressively difficult tasks across three interconnected datasets:

1. **Easy: Data Cleaning** — Clean a users table with mixed-type IDs, heterogeneous date formats, and null corruption
2. **Medium: Join Repair** — Fix silent join failures between users and transactions caused by schema drift, then extract nested JSON metadata into flat columns
3. **Hard: Root Cause Analysis** — Correlate all three tables (users, transactions, logs), filter adversarial noise to find the true signal, and produce a fully reconciled dataset

## The State Space

- **Observation Space**: Agents receive semantic summaries at each step:
  - `dataset_sample`: Top 3 row preview
  - `columns` & `dtypes`: Current schema
  - `missing_values`: Null count per column
  - `total_rows`: Current row count
  - `debug_hints`: Environment-detected anomalies
  - `feedback`: Execution results from previous actions
- **Action Space**: Structured Pydantic actions:
  - `execute_pandas`: Arbitrary Pandas code
  - `merge_tables`: Cross-table joins with type hints
  - `parse_json`: Extract nested JSON into flat columns
  - `cast_type`, `fill_nan`, `drop_column`, `rename_column`, `drop_duplicates`, `submit`

## Installation & Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate datasets:**
   ```bash
   python generate_data.py
   ```

3. **Run Inference:**
   ```bash
   export OPENAI_API_KEY="your-key-here"
   export MODEL_NAME="llama-3.3-70b-versatile"
   export API_BASE_URL="https://api.groq.com/openai/v1"
   export HF_TOKEN="your-hf-token"
   python inference.py
   ```

4. **Run FastAPI server (for HF Spaces / Docker):**
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 7860
   ```

## Baseline Agent Scores

Using the default zero-shot implementation with `llama-3.3-70b-versatile`:
- **Task 1 (Easy):** `0.69` — Data cleaning achieved
- **Task 2 (Medium):** `0.70` — Join repaired, metadata extracted
- **Task 3 (Hard):** `0.55` — Full root cause analysis completed
- **Average:** `0.65`

## Architecture

```
generate_data.py  →  Faker-based deterministic data generation
models.py         →  Pydantic: State, Observation, Action, Reward(value)
environment.py    →  Gym-style: reset(), step(), state()
tasks.py          →  3 tasks with multi-component deterministic graders
inference.py      →  OpenAI client loop with [START]/[STEP]/[END] logging
app.py            →  FastAPI server on port 7860 with /health
```

## Docker Deployment

```bash
docker build -t openenv-data-engineering .
docker run -p 7860:7860 openenv-data-engineering
```

## Hugging Face Spaces

Deploy as a Docker Space. The server exposes `/health` on port 7860 returning HTTP 200.

## Project Structure

```
/data
openenv.yaml
environment.py
models.py
tasks.py
inference.py
generate_data.py
app.py
Dockerfile
requirements.txt
README.md
```