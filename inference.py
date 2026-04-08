import os
import json
from openai import OpenAI

from environment import DataEnv
from tasks import TASKS


def run_inference():
    print("[START] Initialization")

    # Required environment variables (strict format)
    API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
    HF_TOKEN = os.getenv("HF_TOKEN")  # No default - must be provided

    # Get API key from environment
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    # Initialize LLM client using standard OpenAI client
    client = None
    if OPENAI_API_KEY:
        try:
            client = OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)
            print("[INFO] LLM client initialized")
        except Exception as e:
            print(f"[INFO] LLM client init failed: {e}")
            client = None

    env = DataEnv()
    results = {}

    for task_id in TASKS.keys():
        print(f"[STEP] Evaluating {task_id}")
        obs = env.reset(task_id=task_id)

        done = False
        step_count = 0
        final_score = 0.0

        while not done and step_count < 20:
            print(f"[STEP] Iterating (step {step_count + 1}) on {task_id}")

            prompt = f"""
You are a data engineering AI.
Dataset shape: {obs.total_rows} rows. Columns: {obs.columns}
Missing Values: {obs.missing_values}
Task: {obs.task_description}

Feedback from last action: {obs.feedback}
Debug Hints: {obs.debug_hints}
Sample Data: {json.dumps(obs.dataset_sample, default=str)}

Provide JSON representing your action. E.g., {{"action_type": "submit"}} or {{"action_type": "execute_pandas", "code": "df['column'] = ... "}}. Output raw JSON only.
"""

            # Use LLM if client available
            action_dict = {"action_type": "submit"}

            if OPENAI_API_KEY and client:
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"},
                        temperature=0.1,
                    )
                    action_dict = json.loads(response.choices[0].message.content)
                    print(
                        f"[STEP] LLM chose action: {action_dict.get('action_type', 'unknown')}"
                    )
                except Exception as e:
                    print(f"[STEP] LLM error: {e}")
            else:
                # Fallback heuristics when no LLM available
                if task_id == "easy_data_cleaning":
                    if step_count == 0:
                        action_dict = {
                            "action_type": "execute_pandas",
                            "code": "df['user_id'] = pd.to_numeric(df['user_id'].astype(str).str.replace('USR_', ''), errors='coerce').astype('Int64')",
                        }
                    elif step_count == 1:
                        action_dict = {
                            "action_type": "execute_pandas",
                            "code": "df['signup_date'] = pd.to_datetime(df['signup_date'], errors='coerce', format='mixed').dt.strftime('%Y-%m-%d')",
                        }
                    elif step_count == 2:
                        action_dict = {
                            "action_type": "fill_nan",
                            "column_name": "email",
                            "fill_value": "unknown",
                        }
                    elif step_count >= 3:
                        action_dict = {"action_type": "submit"}

                elif task_id == "medium_join_repair":
                    if step_count == 0:
                        action_dict = {
                            "action_type": "execute_pandas",
                            "code": "df['user_id'] = df['user_id'].astype(str).str.replace('USR_', '').astype('Int64')",
                        }
                    elif step_count == 1:
                        action_dict = {
                            "action_type": "merge_tables",
                            "left_on": "user_id",
                            "right_on": "user_id",
                            "how": "left",
                        }
                    elif step_count == 2:
                        action_dict = {
                            "action_type": "parse_json",
                            "column_name": "metadata",
                        }
                    elif step_count == 3:
                        action_dict = {
                            "action_type": "drop_column",
                            "column_name": "metadata",
                        }
                    elif step_count >= 4:
                        action_dict = {"action_type": "submit"}

                elif task_id == "hard_root_cause_analysis":
                    if step_count == 0:
                        action_dict = {
                            "action_type": "execute_pandas",
                            "code": "df['user_id'] = df['user_id'].astype(str).str.replace('USR_', '').astype('Int64')",
                        }
                    elif step_count == 1:
                        action_dict = {
                            "action_type": "merge_tables",
                            "left_on": "user_id",
                            "right_on": "user_id",
                            "how": "left",
                        }
                    elif step_count == 2:
                        action_dict = {
                            "action_type": "parse_json",
                            "column_name": "metadata",
                        }
                    elif step_count == 3:
                        action_dict = {
                            "action_type": "drop_column",
                            "column_name": "metadata",
                        }
                    elif step_count == 4:
                        action_dict = {
                            "action_type": "execute_pandas",
                            "code": "df['signup_date'] = pd.to_datetime(df['signup_date'], errors='coerce', format='mixed').dt.strftime('%Y-%m-%d')",
                        }
                    elif step_count == 5:
                        action_dict = {
                            "action_type": "fill_nan",
                            "column_name": "email",
                            "fill_value": "unknown",
                        }
                    elif step_count == 6:
                        action_dict = {
                            "action_type": "execute_pandas",
                            "code": "df['amount'] = df['amount'].fillna(df['amount'].median())",
                        }
                    elif step_count >= 7:
                        action_dict = {"action_type": "submit"}

            class DummyAction:
                def __init__(self, d):
                    for k, v in d.items():
                        setattr(self, k, v)
                    if not hasattr(self, "action_type"):
                        self.action_type = "submit"

            action = DummyAction(action_dict)
            obs, reward, done, info = env.step(action)
            final_score = reward.value

            step_count += 1

        results[task_id] = final_score
        print(f"[END] Task: {task_id} | Final Score: {final_score:.2f}")

    print("\n[SUMMARY] Results:")
    for tid, score in results.items():
        print(f"  {tid}: {score:.2f}")
    avg = sum(results.values()) / len(results)
    print(f"  Average: {avg:.2f}")


if __name__ == "__main__":
    run_inference()
