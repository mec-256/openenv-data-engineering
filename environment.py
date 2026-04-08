import pandas as pd
import json
import os
from typing import Tuple, Dict, Any

from models import State, Observation, Reward
from tasks import TASKS


class DataEnv:
    def __init__(self):
        self._state = None
        self._task = None
        self._df = None
        self._tables = {}

    def reset(self, task_id: str = "easy_data_cleaning") -> Observation:
        if task_id not in TASKS:
            raise ValueError(f"Task {task_id} not found.")

        self._task = TASKS[task_id]
        self._tables = self._task.load_tables()

        primary_table = self._task.get_primary_table()
        self._df = self._tables[primary_table].copy()

        self._state = State(
            task_id=task_id,
            step_count=0,
            max_steps=20,
            is_done=False,
            cumulative_reward=0.0,
            tables_loaded=[primary_table],
        )

        return self._get_obs(feedback="Environment initialized.")

    def _get_obs(self, feedback: str) -> Observation:
        self._df.columns = self._df.columns.astype(str)

        safe_df = self._df.head(3).where(pd.notnull(self._df.head(3)), None)
        hints = self._task.get_debug_hints(self._df)
        return Observation(
            dataset_sample=safe_df.to_dict(orient="records"),
            columns=list(self._df.columns),
            dtypes={col: str(dtype) for col, dtype in self._df.dtypes.items()},
            total_rows=len(self._df),
            missing_values=self._df.isna().sum().to_dict(),
            task_description=self._task.description,
            feedback=feedback,
            debug_hints=hints,
        )

    def state(self) -> State:
        return self._state

    def step(self, action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        self._state.step_count += 1
        feedback = ""
        reward_value = 0.0

        if self._state.is_done:
            return (
                self._get_obs("Already done."),
                Reward(value=0.0, message="Already done."),
                True,
                {},
            )

        try:
            action_type = getattr(action, "action_type", None) or action.get(
                "action_type"
            )

            if action_type == "submit":
                score, feedback = self._task.grade(self._df, self._tables)
                reward_value = score
                self._state.is_done = True

            elif action_type == "drop_column":
                col = getattr(action, "column_name", None)
                if col in self._df.columns:
                    self._df.drop(columns=[col], inplace=True)
                    feedback = f"Dropped column '{col}'."
                    reward_value = 0.03
                else:
                    feedback = f"Column '{col}' not found."
                    reward_value = -0.05

            elif action_type == "rename_column":
                old_n = getattr(action, "old_name", None)
                new_n = getattr(action, "new_name", None)
                if old_n in self._df.columns:
                    self._df.rename(columns={old_n: new_n}, inplace=True)
                    feedback = f"Renamed '{old_n}' to '{new_n}'."
                    reward_value = 0.02
                else:
                    feedback = f"Column '{old_n}' not found."
                    reward_value = -0.05

            elif action_type == "fill_nan":
                col = getattr(action, "column_name", None)
                val = getattr(action, "fill_value", None)
                if col in self._df.columns:
                    before_na = int(self._df[col].isna().sum())
                    self._df[col] = self._df[col].fillna(val)
                    after_na = int(self._df[col].isna().sum())
                    feedback = f"Filled NaNs in '{col}' with {val}. Reduced missing by {before_na - after_na}."
                    reward_value = 0.03 if after_na < before_na else 0.0
                else:
                    feedback = f"Column '{col}' not found."
                    reward_value = -0.05

            elif action_type == "drop_duplicates":
                subs = getattr(action, "subset", None)
                before = len(self._df)
                self._df.drop_duplicates(subset=subs, inplace=True)
                after = len(self._df)
                dropped = before - after
                feedback = f"Dropped {dropped} duplicate rows."
                reward_value = 0.03 if dropped > 0 else -0.02

            elif action_type == "parse_json":
                col = getattr(action, "column_name", None)
                if col in self._df.columns:

                    def safe_parse(val):
                        try:
                            if isinstance(val, dict):
                                return val
                            return json.loads(val)
                        except Exception:
                            return {}

                    parsed = self._df[col].apply(safe_parse).apply(pd.Series)
                    new_cols = list(parsed.columns)
                    # Drop the original column first to avoid duplicates
                    self._df = self._df.drop(columns=[col])
                    self._df = pd.concat([self._df, parsed], axis=1)
                    feedback = f"Parsed JSON from '{col}' into columns: {new_cols}."
                    reward_value = 0.08
                else:
                    feedback = f"Column '{col}' not found."
                    reward_value = -0.05

            elif action_type == "cast_type":
                col = getattr(action, "column_name", None)
                target = getattr(action, "target_type", None)
                if col in self._df.columns:
                    type_map = {
                        "string": str,
                        "integer": lambda x: pd.to_numeric(x, errors="coerce").astype(
                            "Int64"
                        ),
                        "float": lambda x: pd.to_numeric(x, errors="coerce"),
                        "boolean": lambda x: x.astype(bool),
                        "datetime": lambda x: pd.to_datetime(x, errors="coerce"),
                    }
                    if target in type_map:
                        self._df[col] = type_map[target](self._df[col])
                        feedback = f"Cast '{col}' to {target}."
                        reward_value = 0.05
                    else:
                        feedback = f"Unknown target type: {target}"
                        reward_value = -0.05
                else:
                    feedback = f"Column '{col}' not found."
                    reward_value = -0.05

            elif action_type == "merge_tables":
                left_on = getattr(action, "left_on", None)
                right_on = getattr(action, "right_on", None)
                how = getattr(action, "how", "inner")
                if left_on in self._df.columns:
                    merged = False
                    for tname, tdf in self._tables.items():
                        if (
                            right_on in tdf.columns
                            and tname not in self._state.tables_loaded
                        ):
                            self._df = self._df.merge(
                                tdf, left_on=left_on, right_on=right_on, how=how
                            )
                            self._state.tables_loaded.append(tname)
                            feedback = f"Merged with '{tname}' on {left_on}={right_on} (how={how}). New shape: {len(self._df)} rows."
                            reward_value = 0.1
                            merged = True
                            break
                    if not merged:
                        feedback = (
                            f"No available table has column '{right_on}' to merge on."
                        )
                        reward_value = -0.05
                else:
                    feedback = f"Column '{left_on}' not found in current dataframe."
                    reward_value = -0.05

            elif action_type == "execute_pandas":
                code = getattr(action, "code", None)
                local_vars = {"df": self._df, "pd": pd, "json": json}
                exec(code, {}, local_vars)
                self._df = local_vars["df"]
                feedback = "Successfully executed Pandas code."
                reward_value = 0.02

            else:
                feedback = f"Unknown action type: {action_type}"
                reward_value = -0.1

        except Exception as e:
            feedback = f"Error executing action: {str(e)}"
            reward_value = -0.1

        self._state.cumulative_reward += reward_value

        if self._state.step_count >= self._state.max_steps:
            self._state.is_done = True
            final_score, final_grade_feedback = self._task.grade(self._df, self._tables)
            reward_value = final_score
            feedback += f"\nMax steps reached. Final Grade: {final_grade_feedback}"

        obs = self._get_obs(feedback)
        reward = Reward(value=max(-1.0, min(1.0, reward_value)), message=feedback)

        return obs, reward, self._state.is_done, {}
