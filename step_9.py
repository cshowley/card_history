import os
import pandas as pd
import numpy as np
import constants


def run_step_9():
    print("Starting Step 9: Re-sorting Predictions...")

    input_file = constants.S8_PREDICTIONS_FILE
    output_file = constants.S9_OUTPUT_FILE

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print(f"Reading {input_file}...")
    df = pd.read_parquet(input_file)

    print(f"Loaded {len(df)} rows.")

    sort_cols = ["gemrate_id", "grading_company", "grade", "half_grade"]
    print(f"Sorting dataframe by {sort_cols}...")
    df = df.sort_values(by=sort_cols, ascending=True).reset_index(drop=True)

    group_cols = ["gemrate_id", "grading_company"]
    target_cols = ["prediction_lower", "prediction_upper", "prediction"]

    target_cols = [c for c in target_cols if c in df.columns]
    total_groups = len(df.groupby(group_cols))
    for col in target_cols:
        print(f"  Processing {col}...")

        violation_counts = (
            df.groupby(group_cols)[col]
            .apply(lambda x: np.any(x.values[:-1] > x.values[1:]))
            .sum()
        )
        print(
            f"    Found {violation_counts} / {total_groups} groups violating monotonicity."
        )

        df[col] = df.groupby(group_cols)[col].transform(lambda x: np.sort(x.values))
    
    lower = np.minimum(df["prediction_lower"], df["prediction_upper"])
    upper = np.maximum(df["prediction_lower"], df["prediction_upper"])
    df["prediction_lower"] = lower
    df["prediction_upper"] = upper
    df["prediction"] = (df["prediction_lower"] + df["prediction_upper"]) / 2

    print(f"Saving sorted predictions to {output_file}...")

    df.to_parquet(output_file)

    print("Running QA Spot Check...")

    hist_file = constants.S3_HISTORICAL_DATA_FILE

    print(f"Loading historical data from {hist_file}...")
    hist_df = pd.read_parquet(hist_file)

    hist_df["date"] = pd.to_datetime(hist_df["date"])
    hist_df = hist_df.sort_values("date", ascending=False)

    def get_grading_company(row):
        for col in hist_df.columns:
            if col.startswith("grade_co_") and row[col] == 1:
                return col.replace("grade_co_", "")
        return "Unknown"

    dummy_cols = [c for c in hist_df.columns if c.startswith("grade_co_")]
    if dummy_cols:
        hist_df["grading_company"] = (
            hist_df[dummy_cols].idxmax(axis=1).str.replace("grade_co_", "")
        )

    recent_sales = (
        hist_df.sort_values("date", ascending=False)
        .groupby(["gemrate_id", "grade", "half_grade"])
        .head(1)
        .copy()
    )

    cols_to_keep = [
        "gemrate_id",
        "grade",
        "half_grade",
        "grading_company",
        "price",
        "date",
    ]
    recent_sales = recent_sales[[c for c in cols_to_keep if c in recent_sales.columns]]
    recent_sales = recent_sales.rename(
        columns={"price": "recent_price", "date": "recent_date"}
    )

    merged = pd.merge(
        recent_sales,
        df,
        on=["gemrate_id", "grade", "half_grade", "grading_company"],
        how="inner",
    )

    merged["ratio"] = merged["prediction"] / merged["recent_price"]

    today = pd.to_datetime("today").normalize()
    merged["days_ago"] = (today - merged["recent_date"]).dt.days

    spot_cols = [
        "gemrate_id",
        "grade",
        "half_grade",
        "grading_company",
        "ratio",
        "days_ago",
        "prediction",
        "recent_price",
    ]
    spot_df = merged[spot_cols].rename(columns={"prediction": "predicted_price"})

    spot_file = "spot_check.parquet"
    print(f"Writing {len(spot_df)} spot check rows to {spot_file}...")
    spot_df.to_parquet(spot_file)
    spot_df = spot_df.loc[spot_df["days_ago"] <= 30]
    outliers = spot_df[(spot_df["ratio"] > 1.2) | (spot_df["ratio"] < 0.8)]
    count = len(outliers)
    print(f"Found {count} outliers (prediction / recent_price > 1.2 or < 0.8)")

    print("Done.")


if __name__ == "__main__":
    run_step_9()
