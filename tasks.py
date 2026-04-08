import pandas as pd
import json
import os
from typing import Tuple, Dict, Any


class BaseTask:
    task_id: str
    description: str

    def load_tables(self) -> Dict[str, pd.DataFrame]:
        raise NotImplementedError

    def get_primary_table(self) -> str:
        raise NotImplementedError

    def get_debug_hints(self, df: pd.DataFrame) -> list:
        return []

    def grade(
        self, df: pd.DataFrame, tables: Dict[str, pd.DataFrame]
    ) -> Tuple[float, str]:
        raise NotImplementedError


class DataCleaningTask(BaseTask):
    """
    Easy: Clean users.csv
    - Standardize signup_date to YYYY-MM-DD
    - Handle null emails (mark as 'unknown' or drop)
    - Normalize user_id to integers
    """

    task_id = "easy_data_cleaning"
    description = "Clean users.csv: standardize signup_date to YYYY-MM-DD, fill null emails with 'unknown', normalize user_id to integers."

    def load_tables(self):
        users = pd.read_csv(os.path.join("data", "users.csv"))
        return {"users": users}

    def get_primary_table(self):
        return "users"

    def get_debug_hints(self, df):
        hints = []
        if "user_id" in df.columns:
            mixed = df["user_id"].apply(type).nunique() > 1
            if mixed:
                hints.append("user_id has mixed types (int and str)")
        if "signup_date" in df.columns:
            hints.append(
                f"signup_date has {df['signup_date'].nunique()} unique formats"
            )
        if "email" in df.columns:
            nulls = int(df["email"].isna().sum())
            if nulls > 0:
                hints.append(f"email has {nulls} null values")
        return hints

    def grade(
        self, df: pd.DataFrame, tables: Dict[str, pd.DataFrame]
    ) -> Tuple[float, str]:
        score = 0.0
        feedback_parts = []

        if "signup_date" in df.columns:
            try:
                parsed = pd.to_datetime(df["signup_date"], errors="coerce")
                valid = parsed.dt.strftime("%Y-%m-%d")
                expected = pd.to_datetime(
                    tables["users"]["signup_date"], errors="coerce"
                ).dt.strftime("%Y-%m-%d")
                matches = (valid == expected).sum()
                date_score = matches / len(expected)
                score += date_score * 0.4
                feedback_parts.append(f"Date standardization: {date_score:.2f}")
            except Exception:
                feedback_parts.append("Date standardization failed")
        else:
            feedback_parts.append("Missing signup_date column")

        if "email" in df.columns:
            nulls_before = tables["users"]["email"].isna().sum()
            nulls_after = df["email"].isna().sum()
            if nulls_after == 0:
                score += 0.3
                feedback_parts.append("All null emails handled")
            elif nulls_after < nulls_before:
                score += 0.15
                feedback_parts.append(
                    f"Reduced email nulls from {nulls_before} to {nulls_after}"
                )
            else:
                feedback_parts.append("Email nulls not addressed")
        else:
            feedback_parts.append("Missing email column")

        if "user_id" in df.columns:
            try:
                converted = pd.to_numeric(df["user_id"], errors="coerce")
                valid_ids = converted.notna().sum()
                id_score = valid_ids / len(df)
                score += id_score * 0.3
                feedback_parts.append(f"User ID normalization: {id_score:.2f}")
            except Exception:
                feedback_parts.append("User ID normalization failed")
        else:
            feedback_parts.append("Missing user_id column")

        return min(score, 1.0), "; ".join(feedback_parts)


class JoinRepairTask(BaseTask):
    """
    Medium: Fix join between users and transactions
    - Normalize user_id formats across both tables
    - Merge the tables
    - Extract metadata JSON into flat columns
    """

    task_id = "medium_join_repair"
    description = "Fix user_id mismatch between users and transactions, merge tables, and extract metadata JSON into flat columns (payment_method, location, device)."

    def load_tables(self):
        users = pd.read_csv(os.path.join("data", "users.csv"))
        transactions = pd.read_csv(os.path.join("data", "transactions.csv"))
        return {"users": users, "transactions": transactions}

    def get_primary_table(self):
        return "transactions"

    def get_debug_hints(self, df):
        hints = []
        if "user_id" in df.columns:
            str_count = df["user_id"].astype(str).str.startswith("USR_").sum()
            int_count = (~df["user_id"].astype(str).str.startswith("USR_")).sum()
            hints.append(f"user_id: {int_count} int-like, {str_count} USR_ prefixed")
        if "metadata" in df.columns:
            is_str = df["metadata"].apply(lambda x: isinstance(x, str)).sum()
            hints.append(f"metadata: {is_str} stringified JSON entries")
        return hints

    def grade(
        self, df: pd.DataFrame, tables: Dict[str, pd.DataFrame]
    ) -> Tuple[float, str]:
        score = 0.0
        feedback_parts = []

        merged = len(df) > len(tables["transactions"])
        if merged:
            score += 0.3
            feedback_parts.append("Tables merged successfully")
        else:
            feedback_parts.append("Tables not merged")

        if "payment_method" in df.columns:
            non_null = int(df["payment_method"].notna().sum())
            score += min(0.2, float(non_null / len(df)) * 0.2)
            feedback_parts.append(f"payment_method extracted: {non_null} rows")
        else:
            feedback_parts.append("payment_method not extracted")

        if "location" in df.columns:
            non_null = int(df["location"].notna().sum())
            score += min(0.2, float(non_null / len(df)) * 0.2)
            feedback_parts.append(f"location extracted: {non_null} rows")
        else:
            feedback_parts.append("location not extracted")

        if "device" in df.columns:
            non_null = int(df["device"].notna().sum())
            score += min(0.15, float(non_null / len(df)) * 0.15)
            feedback_parts.append(f"device extracted: {non_null} rows")
        else:
            feedback_parts.append("device not extracted")

        if "metadata" not in df.columns:
            score += 0.15
            feedback_parts.append("Original metadata column dropped")

        return min(score, 1.0), "; ".join(feedback_parts)


class RootCauseAnalysisTask(BaseTask):
    """
    Hard: Correlate all 3 tables, filter log noise, find true signal
    - Identify user ID format mismatch as root cause
    - Produce clean joined dataset with all fields
    - Extract metadata, standardize dates, handle nulls
    """

    task_id = "hard_root_cause_analysis"
    description = "Correlate users, transactions, and logs. Identify root cause of data corruption (user ID format mismatch). Produce a clean joined dataset with standardized dates, extracted metadata, and handled nulls."

    def load_tables(self):
        users = pd.read_csv(os.path.join("data", "users.csv"))
        transactions = pd.read_csv(os.path.join("data", "transactions.csv"))
        logs = pd.read_csv(os.path.join("data", "logs.csv"))
        return {"users": users, "transactions": transactions, "logs": logs}

    def get_primary_table(self):
        return "transactions"

    def get_debug_hints(self, df):
        hints = []
        if "user_id" in df.columns:
            str_count = df["user_id"].astype(str).str.startswith("USR_").sum()
            if str_count > 0:
                hints.append(f"WARNING: {str_count} rows have USR_ prefixed user_ids")
        if "level" in df.columns and "message" in df.columns:
            mismatch_logs = df[
                df["message"].str.contains("mismatch", case=False, na=False)
            ]
            if len(mismatch_logs) > 0:
                hints.append(
                    f"Found {len(mismatch_logs)} log entries about ID mismatch"
                )
        return hints

    def grade(
        self, df: pd.DataFrame, tables: Dict[str, pd.DataFrame]
    ) -> Tuple[float, str]:
        score = 0.0
        feedback_parts = []

        has_users_cols = {"name", "email", "signup_date"}.issubset(set(df.columns))
        if has_users_cols:
            score += 0.2
            feedback_parts.append("User columns present (tables joined)")
        else:
            feedback_parts.append("User columns missing (tables not joined)")

        has_metadata_cols = {"payment_method", "location", "device"}.issubset(
            set(df.columns)
        )
        if has_metadata_cols:
            score += 0.2
            feedback_parts.append("Metadata extracted into flat columns")
        else:
            found = {"payment_method", "location", "device"}.intersection(
                set(df.columns)
            )
            score += len(found) * 0.06
            feedback_parts.append(f"Metadata partially extracted: {found}")

        if "metadata" not in df.columns:
            score += 0.1
            feedback_parts.append("Original metadata column removed")

        if "signup_date" in df.columns:
            try:
                parsed = pd.to_datetime(df["signup_date"], errors="coerce")
                valid = parsed.notna().sum()
                date_score = valid / len(df)
                score += date_score * 0.15
                feedback_parts.append(f"Dates standardized: {date_score:.2f}")
            except Exception:
                feedback_parts.append("Date standardization failed")

        if "email" in df.columns:
            nulls = df["email"].isna().sum()
            if nulls == 0:
                score += 0.1
                feedback_parts.append("All email nulls handled")
            else:
                feedback_parts.append(f"Email still has {nulls} nulls")

        if "user_id" in df.columns:
            try:
                converted = pd.to_numeric(df["user_id"], errors="coerce")
                valid = converted.notna().sum()
                id_score = valid / len(df)
                score += id_score * 0.15
                feedback_parts.append(f"User IDs normalized: {id_score:.2f}")
            except Exception:
                feedback_parts.append("User ID normalization failed")

        if "amount" in df.columns:
            nulls = df["amount"].isna().sum()
            if nulls == 0:
                score += 0.1
                feedback_parts.append("All amount nulls handled")
            else:
                feedback_parts.append(f"Amount still has {nulls} nulls")

        return min(score, 1.0), "; ".join(feedback_parts)


TASKS = {
    DataCleaningTask.task_id: DataCleaningTask(),
    JoinRepairTask.task_id: JoinRepairTask(),
    RootCauseAnalysisTask.task_id: RootCauseAnalysisTask(),
}
