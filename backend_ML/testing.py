import os
import sys
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Amount thresholds
HIGH_VALUE = 20_000

# Time windows
ODD_HOUR_START, ODD_HOUR_END = 0, 4        # inclusive (00:00–04:59)
STRUCT_WINDOW_HOURS = 24                   # window for structuring per (from->to)
VELOCITY_WINDOW_HOURS = 24                 # outgoing velocity per originating account
DIVERSITY_WINDOW_DAYS = 1                  # distinct counterparties per day

# Counts / thresholds
STRUCT_MIN_COUNT = 5                       # >= N small txns in window to same counterparty
VELOCITY_MIN_COUNT = 20                    # >= N outgoing txns in window from same account
COUNTERPARTY_DIVERSITY_MIN = 10            # >= N distinct counterparties per day


def load_new_dataset(filepath):
    

    if not os.path.exists(filepath):
        print(f"ERROR: CSV file not found at path: {filepath}", file=sys.stderr)
        sys.exit(1)

    raw = pd.read_csv(filepath)

    y = raw["Is Laundering"].astype(int) #for checking later


    raw.columns = [c.strip() for c in raw.columns]

    # Map first two "Account*" columns to from/to account
    cols = raw.columns.tolist()
    acct_like = [i for i, c in enumerate(cols) if c.replace(".", "").lower().startswith("account")]
    if len(acct_like) >= 2:
        cols[acct_like[0]] = "from_account"
        cols[acct_like[1]] = "to_account"
        raw.columns = cols

    # Standardize common headers if present
    name_map = {
        "Timestamp": "timestamp",
        "From Bank": "from_bank",
        "To Bank": "to_bank",
        "Amount Received": "amount_received",
        "Receiving Currency": "receiving_currency",
        "Amount Paid": "amount_paid",
        "Payment Currency": "payment_currency",
        "Payment Format": "payment_format",
        "Is Laundering": "is_laundering",   # not used here, but we won’t break if present
    }
    for k, v in name_map.items():
        if k in raw.columns:
            raw = raw.rename(columns={k: v})

    df = raw  # no extra copy

    # Ensure key columns exist
    needed_cols = [
        "timestamp","from_bank","to_bank","from_account","to_account",
        "amount_received","amount_paid","payment_currency","receiving_currency",
        "payment_format"
    ]
    for col in needed_cols:
        if col not in df.columns:
            df[col] = np.nan if col in ["amount_received", "amount_paid"] else ""

    # Parse timestamp & numeric amounts
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    for amt in ["amount_received", "amount_paid"]:
        s = df[amt].astype(str).str.replace(",", "", regex=False).str.strip()
        df[amt] = pd.to_numeric(s.replace({"": np.nan}), errors="coerce").astype("float32")

    # Use categories for low-cardinality text to save memory
    for c in ["from_bank","to_bank","payment_currency","receiving_currency","payment_format"]:
        if c in df.columns:
            df[c] = df[c].astype("category")

    # Choose canonical amount column
    amount_col = "amount_paid" if df["amount_paid"].notna().any() else "amount_received"
    amt = df[amount_col].fillna(0.0).astype("float32")

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # -------------------------
    # 2) SIMPLE FLAGS (O(N), vectorized)
    # -------------------------
    

    df["flag_high_value"]   = (amt > HIGH_VALUE)
    df["flag_cross_bank"]   = (df["from_bank"].astype(str).values != df["to_bank"].astype(str).values)
    df["flag_same_account"] = (df["from_account"].astype(str).values == df["to_account"].astype(str).values)

    df["hour"] = df["timestamp"].dt.hour
    df["flag_odd_hours"] = df["hour"].between(ODD_HOUR_START, ODD_HOUR_END, inclusive="both")

    if {"payment_currency","receiving_currency"}.issubset(df.columns):
        df["flag_cross_currency"] = (
            df["payment_currency"].astype(str).str.upper().values
            != df["receiving_currency"].astype(str).str.upper().values
        )
    else:
        df["flag_cross_currency"] = False

    # Round-amount heuristic (suspiciously tidy amounts, tune as needed)
    # e.g., multiples of 1000, or ending with .00
    df["flag_round_amount"] = (np.isclose((amt % 1000), 0) | np.isclose((amt * 100) % 100, 0)).astype(bool)

    # -------------------------
    # 3) TIME-WINDOW PATTERNS via downsampling (fast & scalable)
    # -------------------------
    

    # Hour and Day buckets
    df["ts_hour"] = df["timestamp"].dt.floor("H")
    df["ts_day"]  = df["timestamp"].dt.floor("D")

    # Guard on timestamps
    have_time = df["ts_hour"].notna().any()
    if not have_time:
        # If no timestamps, skip time-based flags safely
        df["flag_structuring"] = False
        df["flag_velocity_outgoing"] = False
        df["flag_counterparty_diversity"] = False
    else:
        # --- 3a) Structuring (many small transfers to same counterparty in 24h)
        small = amt < HIGH_VALUE
        small_df = df.loc[small, ["from_account","to_account","ts_hour"]].dropna(subset=["ts_hour"])

        # Hourly counts per (from->to)
        hourly_ft = (
            small_df
            .groupby(["from_account","to_account","ts_hour"], sort=True)
            .size().rename("cnt").reset_index()
        )

        # Rolling 24h sum over hourly counts per pair
        hourly_ft = hourly_ft.sort_values(["from_account","to_account","ts_hour"]).set_index("ts_hour")
        rolling_counts = (
            hourly_ft
            .groupby(["from_account","to_account"])["cnt"]
            .rolling(f"{STRUCT_WINDOW_HOURS}H")
            .sum()
            .reset_index()
        )
        rolling_counts["struct_hit"] = rolling_counts["cnt"] >= STRUCT_MIN_COUNT

        # Merge back by hour bucket
        df = df.merge(
            rolling_counts[["from_account","to_account","ts_hour","struct_hit"]],
            on=["from_account","to_account","ts_hour"],
            how="left"
        )
        df["flag_structuring"] = df["struct_hit"].fillna(False).values
        df.drop(columns=["struct_hit"], inplace=True)

        # --- 3b) Velocity: many outgoing txns from same origin account within 24h
        hourly_from = (
            df.dropna(subset=["ts_hour"])
              .groupby(["from_account","ts_hour"], sort=True)
              .size().rename("out_cnt").reset_index()
        )
        hourly_from = hourly_from.sort_values(["from_account","ts_hour"]).set_index("ts_hour")
        vel_roll = (
            hourly_from
            .groupby(["from_account"])["out_cnt"]
            .rolling(f"{VELOCITY_WINDOW_HOURS}H").sum()
            .reset_index()
        )
        vel_roll["vel_hit"] = vel_roll["out_cnt"] >= VELOCITY_MIN_COUNT

        df = df.merge(
            vel_roll[["from_account","ts_hour","vel_hit"]],
            on=["from_account","ts_hour"],
            how="left"
        )
        df["flag_velocity_outgoing"] = df["vel_hit"].fillna(False).values
        df.drop(columns=["vel_hit"], inplace=True)

        # --- 3c) Counterparty diversity per day: many distinct to_accounts from same origin
        daily_div = (
            df.dropna(subset=["ts_day"])
              .groupby(["from_account","ts_day"])["to_account"]
              .nunique()
              .rename("uniq_to")
              .reset_index()
        )
        daily_div["div_hit"] = daily_div["uniq_to"] >= COUNTERPARTY_DIVERSITY_MIN

        df = df.merge(
            daily_div[["from_account","ts_day","div_hit"]],
            on=["from_account","ts_day"],
            how="left"
        )
        df["flag_counterparty_diversity"] = df["div_hit"].fillna(False).values
        df.drop(columns=["div_hit"], inplace=True)

    # -------------------------
    # 4) SCORING & SEVERITY
    # -------------------------
    
    # Choose which flags to score (tune weights per policy)
    rule_cols = [
        "flag_high_value","flag_cross_bank","flag_same_account",
        "flag_odd_hours","flag_cross_currency","flag_round_amount",
        "flag_structuring","flag_velocity_outgoing","flag_counterparty_diversity"
    ]

    X = df[rule_cols].fillna(0).astype(int)
    laundering_count = (df["is_laundering"] == 1).sum()
    num_rows = df.shape[0]


    return (X, laundering_count, num_rows, df)
