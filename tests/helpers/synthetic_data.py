"""
Factory functions for building controlled synthetic DataFrames.

Each factory creates data with known properties so that metric
computations in the pipeline steps produce predictable, verifiable results.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Step 1 helpers
# ---------------------------------------------------------------------------

def make_ebay_df(
    n_rows=10,
    n_before_cutoff=3,
    n_zero_bids=2,
    n_single_bids=1,
    n_missing_gemrate_id=1,
    n_missing_grade=1,
    n_missing_price=1,
    n_extreme_low_price=1,
    n_extreme_high_price=1,
):
    """Build a synthetic eBay-shaped DataFrame with controlled anomalies.

    Returns the DataFrame and a dict of expected metric values.
    """
    rng = np.random.default_rng(42)

    # Dates: n_before_cutoff rows before 2025-09-01, rest after
    cutoff = datetime(2025, 9, 1)
    dates_before = [
        (cutoff - timedelta(days=i + 1)).strftime("%Y-%m-%dT%H:%M:%S")
        for i in range(n_before_cutoff)
    ]
    dates_after = [
        (cutoff + timedelta(days=i + 1)).strftime("%Y-%m-%dT%H:%M:%S")
        for i in range(n_rows - n_before_cutoff)
    ]
    all_dates = dates_before + dates_after

    # Prices: mix of normal, extreme low, extreme high
    prices = []
    for i in range(n_rows):
        if i < n_extreme_low_price:
            prices.append("US $0.001")
        elif i < n_extreme_low_price + n_extreme_high_price:
            prices.append("US $150000.00")
        elif i < n_extreme_low_price + n_extreme_high_price + n_missing_price:
            prices.append(np.nan)
        else:
            prices.append(f"US ${rng.uniform(10, 500):.2f}")

    # Bids
    bids = [5] * n_rows
    for i in range(n_zero_bids):
        bids[i] = 0
    for i in range(n_zero_bids, n_zero_bids + n_single_bids):
        bids[i] = 1

    # Gemrate IDs and grades
    gemrate_ids = [f"GEM_{i:04d}" for i in range(n_rows)]
    grades = [str(rng.choice([7, 8, 9, 10])) for _ in range(n_rows)]

    # Inject missing values
    for i in range(n_missing_gemrate_id):
        gemrate_ids[n_rows - 1 - i] = ""
    for i in range(n_missing_grade):
        grades[n_rows - 1 - i] = np.nan

    df = pd.DataFrame({
        "item_data.date": all_dates,
        "item_data.price": prices,
        "item_data.number_of_bids": bids,
        "gemrate_data.gemrate_id": gemrate_ids,
        "gemrate_data.grade": grades,
    })

    # Expected values (after cutoff filtering, the before-cutoff rows are dropped)
    n_after = n_rows - n_before_cutoff

    # Extreme prices counted across ALL rows (before cutoff filtering)
    # but the step computes them AFTER filtering, so only count in the after-cutoff rows.
    # Indices 0..n_before_cutoff-1 are before-cutoff. extreme_low is idx 0, extreme_high is idx 1.
    # After filtering, only indices n_before_cutoff..n_rows-1 survive.
    # extreme_low at idx 0 is dropped (before cutoff), extreme_high at idx 1 is dropped (before cutoff if n_before_cutoff >= 2).
    # We need to be careful: extremes are placed at indices 0, 1, ... which overlap with before-cutoff.
    # Let's re-think: the extreme prices are counted AFTER cutoff filtering in step_1.
    # Since we place extremes at indices 0,1 and before-cutoff at indices 0..n_before_cutoff-1,
    # when n_before_cutoff >= n_extreme_low_price + n_extreme_high_price, all extremes are dropped.
    # Default: n_before_cutoff=3, n_extreme_low=1, n_extreme_high=1, so both extremes are in before-cutoff.

    expected = {
        "s1_sales_before_sep_2025": n_before_cutoff,
        "s1_dropped_sales_before_sep_2025": n_before_cutoff,
        "s1_ebay_records": n_after,
        "s1_zero_bids": n_zero_bids,
        "s1_single_bids": n_single_bids,
    }

    return df, expected


def make_pwcc_df(
    n_rows=5,
    n_before_cutoff=1,
    n_missing_gemrate_id=1,
    n_missing_grade=0,
    n_missing_price=0,
    n_extreme_low_price=0,
    n_extreme_high_price=0,
):
    """Build a synthetic PWCC-shaped DataFrame with controlled anomalies."""
    cutoff = datetime(2025, 9, 1)

    dates_before = [
        (cutoff - timedelta(days=i + 1)).strftime("%Y-%m-%dT%H:%M:%S EST")
        for i in range(n_before_cutoff)
    ]
    dates_after = [
        (cutoff + timedelta(days=i + 1)).strftime("%Y-%m-%dT%H:%M:%S EST")
        for i in range(n_rows - n_before_cutoff)
    ]
    all_dates = dates_before + dates_after

    prices = []
    for i in range(n_rows):
        if i < n_extreme_low_price:
            prices.append(0.005)
        elif i < n_extreme_low_price + n_extreme_high_price:
            prices.append(200000.0)
        elif i < n_extreme_low_price + n_extreme_high_price + n_missing_price:
            prices.append(np.nan)
        else:
            prices.append(round(np.random.uniform(10, 500), 2))

    gemrate_ids = [f"GEM_{i:04d}" for i in range(100, 100 + n_rows)]
    grades = [str(np.random.choice([7, 8, 9, 10])) for _ in range(n_rows)]

    for i in range(n_missing_gemrate_id):
        gemrate_ids[n_rows - 1 - i] = ""

    df = pd.DataFrame({
        "api_response.soldDate": all_dates,
        "api_response.purchasePrice": prices,
        "gemrate_data.gemrate_id": gemrate_ids,
        "gemrate_data.grade": grades,
    })

    n_after = n_rows - n_before_cutoff
    expected = {
        "pwcc_before_cutoff": n_before_cutoff,
        "s1_pwcc_records": n_after,
    }

    return df, expected


# ---------------------------------------------------------------------------
# Step 2 helpers
# ---------------------------------------------------------------------------

def make_catalog_csv(path, n_rows=5, n_empty_text=1, n_duplicates=1):
    """Write a synthetic catalog CSV and return expected metric values."""
    rows = []
    for i in range(n_rows + n_duplicates):
        gid = f"GEM_{i:04d}" if i < n_rows else f"GEM_{0:04d}"
        if i < n_empty_text:
            # Use whitespace-only values for text fields so they survive CSV
            # round-trip as non-NaN but produce empty text after strip().
            # Bare "" in CSV is read back as NaN by pandas, which astype(str)
            # converts to the literal string "nan" before fillna can catch it.
            # UNIVERSAL_GEMRATE_ID must be NaN (empty) so step 2 doesn't
            # replace GEMRATE_ID with a space.
            rows.append({
                "GEMRATE_ID": gid,
                "UNIVERSAL_GEMRATE_ID": "",
                "YEAR": " ",
                "SET_NAME": " ",
                "NAME": " ",
                "CARD_NUMBER": " ",
                "PARALLEL": " ",
            })
        else:
            rows.append({
                "GEMRATE_ID": gid,
                "UNIVERSAL_GEMRATE_ID": "",
                "YEAR": "2023",
                "SET_NAME": f"Set {i}",
                "NAME": f"Pokemon {i}",
                "CARD_NUMBER": str(i),
                "PARALLEL": "Holo" if i % 2 == 0 else "",
            })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)

    expected = {
        "s2_input_rows": n_rows,  # after dedup
        "s2_empty_text_count": n_empty_text,
    }
    return expected


# ---------------------------------------------------------------------------
# Step 9 helpers
# ---------------------------------------------------------------------------

def make_predictions_df(n_groups=4, n_grades=4, n_violations=2):
    """Build predictions with controlled monotonicity violations.

    Args:
        n_groups: Number of (gemrate_id, grading_company) groups.
        n_grades: Number of grades per group (7, 8, 9, 10).
        n_violations: How many groups should have monotonicity violations
                      in the prediction column. Must be <= n_groups.

    Returns:
        DataFrame and expected metric dict.
    """
    rows = []
    grades = list(range(7, 7 + n_grades))

    for g in range(n_groups):
        gid = f"GEM_{g:04d}"
        company = "PSA" if g % 2 == 0 else "BGS"

        for grade_idx, grade in enumerate(grades):
            # Base prediction increases with grade (monotonic)
            base = 10.0 + grade_idx * 5.0

            if g < n_violations:
                # Inject violation: make a higher grade cheaper than lower
                # Reverse the last two grades' predictions
                if grade_idx == n_grades - 1:
                    pred = base - 15.0  # Much lower than expected
                else:
                    pred = base
            else:
                pred = base

            rows.append({
                "gemrate_id": gid,
                "grading_company": company,
                "grade": float(grade),
                "half_grade": 0.0,
                "prediction": pred,
                "prediction_lower": pred * 0.8,
                "prediction_upper": pred * 1.2,
            })

    df = pd.DataFrame(rows)

    expected = {
        "s9_total_groups": n_groups,
        "s9_monotonicity_violations_prediction": n_violations,
        "s9_post_sort_violations": 0,
    }
    return df, expected


def make_historical_for_qa(predictions_df, n_outliers=2, n_within_30_days=None):
    """Build historical data that will produce a known number of QA outliers.

    For each unique (gemrate_id, grade, half_grade) in predictions_df, create
    one recent sale. Control the price ratio to produce exactly n_outliers.

    Args:
        predictions_df: The predictions DataFrame (after monotonicity sort).
        n_outliers: How many rows should have ratio outside [0.8, 1.2].
        n_within_30_days: If None, all rows are within 30 days.
    """
    unique_combos = predictions_df.drop_duplicates(
        subset=["gemrate_id", "grade", "half_grade", "grading_company"]
    ).copy()

    today = pd.to_datetime("today").normalize()
    rows = []
    outlier_count = 0

    # Build grading company dummy columns
    companies = unique_combos["grading_company"].unique()

    for _, row in unique_combos.iterrows():
        pred_val = row["prediction"]

        if outlier_count < n_outliers:
            # Price that makes ratio > 1.2 (prediction much higher than price)
            recent_price = pred_val / 1.5  # ratio = 1.5
            outlier_count += 1
        else:
            # Price that makes ratio ~1.0 (within tolerance)
            recent_price = pred_val

        hist_row = {
            "gemrate_id": row["gemrate_id"],
            "grade": row["grade"],
            "half_grade": row["half_grade"],
            "price": recent_price,
            "date": today - timedelta(days=5),  # within 30 days
        }
        # Add grading company dummies
        for c in companies:
            hist_row[f"grade_co_{c}"] = 1 if row["grading_company"] == c else 0

        rows.append(hist_row)

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Step 8 helpers
# ---------------------------------------------------------------------------

def make_today_data_parquet(path, n_rows=20, feature_cols=None):
    """Write a synthetic today_data_with_neighbors parquet file.

    Returns expected prediction-related info.
    """
    if feature_cols is None:
        feature_cols = [
            "grade", "half_grade", "seller_popularity",
            "grade_co_PSA", "grade_co_BGS", "grade_co_CGC",
            "prev_1_price", "prev_1_days_ago",
        ]

    rng = np.random.default_rng(99)
    data = {"gemrate_id": [f"GEM_{i:04d}" for i in range(n_rows)]}

    for col in feature_cols:
        if col.startswith("grade_co_"):
            data[col] = rng.choice([0, 1], size=n_rows)
        elif col == "grade":
            data[col] = rng.choice([7.0, 8.0, 9.0, 10.0], size=n_rows)
        elif col == "half_grade":
            data[col] = rng.choice([0.0, 0.5], size=n_rows)
        else:
            data[col] = rng.uniform(0, 100, size=n_rows)

    df = pd.DataFrame(data)
    df.to_parquet(path, index=False)
    return df
