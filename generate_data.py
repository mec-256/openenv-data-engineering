"""
Generate advanced, multi-table corrupted datasets for the OpenEnv environment.
Includes schema drift, join corruption, nested JSON corruption, and misleading logs.
Uses Faker for deterministic, reproducible data.
"""

import os
import csv
import json
import random
from datetime import datetime, timedelta

from faker import Faker

SEED = 42
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
NUM_ROWS = 500

os.makedirs(DATA_DIR, exist_ok=True)


def generate_users():
    """
    Users dataset with:
    - Schema drift: user_id as int vs string (30% drift)
    - Null corruption: 10% missing emails
    - Date format inconsistencies: 3 different formats
    """
    fake = Faker("en_US")
    random.seed(SEED)
    Faker.seed(SEED)

    users = []
    for i in range(NUM_ROWS):
        user_id = i if random.random() > 0.3 else f"USR_{i}"

        users.append(
            {
                "user_id": user_id,
                "name": fake.name(),
                "email": fake.email() if random.random() > 0.1 else None,
                "signup_date": fake.date_this_decade().strftime(
                    random.choice(["%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y"])
                ),
                "status": random.choice(["active", "inactive", "suspended"]),
            }
        )

    fpath = os.path.join(DATA_DIR, "users.csv")
    with open(fpath, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["user_id", "name", "email", "signup_date", "status"]
        )
        writer.writeheader()
        writer.writerows(users)
    print(f"[DATA] Generated {fpath} ({len(users)} rows)")
    return users


def generate_transactions(users):
    """
    Transactions dataset with:
    - Join corruption: 40% user_id mismatch (int vs USR_x format)
    - Nested JSON corruption: metadata always stored as JSON string
    - Null amounts: 10% missing
    - Silent join failure risk
    """
    fake = Faker("en_US")
    random.seed(SEED + 1)
    Faker.seed(SEED + 1)

    transactions = []
    for i in range(NUM_ROWS):
        user_ref = random.randint(0, NUM_ROWS - 1)

        if random.random() < 0.4:
            user_ref = f"USR_{user_ref}"

        amount = round(random.uniform(10, 500), 2)

        metadata = {
            "payment_method": random.choice(["card", "upi", "bank"]),
            "location": fake.city(),
            "device": random.choice(["mobile", "desktop", "tablet"]),
        }

        # Always serialize metadata as JSON string
        metadata = json.dumps(metadata)

        if random.random() < 0.1:
            amount = None

        transactions.append(
            {
                "transaction_id": i,
                "user_id": user_ref,
                "amount": amount,
                "metadata": metadata,
                "timestamp": (
                    datetime(2024, 1, 1) + timedelta(minutes=random.randint(0, 525600))
                ).isoformat(),
            }
        )

    fpath = os.path.join(DATA_DIR, "transactions.csv")
    with open(fpath, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["transaction_id", "user_id", "amount", "metadata", "timestamp"],
        )
        writer.writeheader()
        writer.writerows(transactions)
    print(f"[DATA] Generated {fpath} ({len(transactions)} rows)")
    return transactions


def generate_logs():
    """
    Logs dataset with:
    - Misleading noise (common false signals)
    - Rare true signal (schema drift warning, only 5 occurrences)
    - Tests agent ability to filter signal from noise
    """
    random.seed(SEED + 2)

    logs = []
    noise_messages = [
        "Null values detected in transactions.amount",
        "Data pipeline executed successfully",
        "Minor delay in ingestion",
        "Schema validation passed",
        "Cache refresh completed",
        "Rate limit threshold approaching",
        "Connection pool utilization at 60%",
    ]

    for _ in range(200):
        logs.append(
            {
                "level": random.choice(["INFO", "WARNING", "ERROR"]),
                "message": random.choice(noise_messages),
                "timestamp": (
                    datetime(2024, 1, 1) + timedelta(minutes=random.randint(0, 525600))
                ).isoformat(),
            }
        )

    for _ in range(5):
        logs.append(
            {
                "level": "WARNING",
                "message": "User ID format mismatch detected between tables",
                "timestamp": (
                    datetime(2024, 6, 1) + timedelta(minutes=random.randint(0, 525600))
                ).isoformat(),
            }
        )

    random.shuffle(logs)

    fpath = os.path.join(DATA_DIR, "logs.csv")
    with open(fpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["level", "message", "timestamp"])
        writer.writeheader()
        writer.writerows(logs)
    print(f"[DATA] Generated {fpath} ({len(logs)} rows)")


if __name__ == "__main__":
    print("[START] Generating advanced datasets...")
    users = generate_users()
    generate_transactions(users)
    generate_logs()
    print("[END] All datasets generated.")
