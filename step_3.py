import os
import re
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

import constants
from data_integrity import get_tracker


def extract_numbers(text):
    text = text.replace(",", "")
    pattern = r"-?(?:\d+(?:\.\d*)?|\.\d+)"
    numbers = re.findall(pattern, text)
    return [float(num) for num in numbers][0]


def s3_clean_grade(val):
    s = str(val).lower().strip().replace("g", "").replace("_", ".")
    if s in ["nan", "none", "", "0", "auth"]:
        return np.nan

    if "10b" in s or "10black" in s:
        return 11.0

    if any(x in s for x in ["pristine", "perfect", "10p", "gem", "mint"]):
        return 10.5

    match = re.search(r"(\d+(\.\d+)?)", s)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return np.nan
    return np.nan


def s3_process_grade(val):
    if pd.isna(val):
        return np.nan, np.nan
    floor_val = np.floor(val)
    half_val = 1.0 if (val - floor_val) > 0 else 0.0
    return floor_val, half_val


def s3_group_currencies(val):
    s = str(val).strip()
    if not s:
        return "Unknown"
    if s.startswith("$") or s[0].isdigit():
        return "$ (No Country Code)"
    return s


def s3_calculate_seller_popularity(df):
    df = df.copy()
    total_sales = len(df)
    seller_counts = df["seller"].value_counts()
    df["seller_popularity"] = df["seller"].map(seller_counts) / total_sales
    df["seller_popularity"] = df["seller_popularity"].fillna(0)
    df = df.drop(columns=["seller"])
    return df


def s3_create_previous_sale_features(df, n_sales_back):
    index_series = None
    idx_df = pd.read_csv(constants.S1_INDEX_FILE)
    idx_df["date"] = pd.to_datetime(idx_df["date"])
    idx_df = idx_df.groupby("date")["index_value"].mean()
    index_series = idx_df
    df = df.sort_values(["gemrate_id", "grade", "date"]).reset_index(drop=True)

    feature_cols = [
        "price",
        "half_grade",
        "grade_co_BGS",
        "grade_co_CGC",
        "grade_co_PSA",
    ]

    feature_cols = [col for col in feature_cols if col in df.columns]
    new_columns = []

    new_data = {}

    for n in range(1, n_sales_back + 1):
        prefix = f"prev_{n}"
        for col in feature_cols:
            new_col = f"{prefix}_{col}"
            new_data[new_col] = df.groupby(["gemrate_id", "grade"])[col].shift(n)
            new_columns.append(new_col)

        days_col = f"{prefix}_days_ago"
        prev_date = df.groupby(["gemrate_id", "grade"])["date"].shift(n)
        new_data[days_col] = (df["date"] - prev_date).dt.days
        new_columns.append(days_col)

        idx_col = f"{prefix}_index_price"
        if index_series is not None:
            new_data[idx_col] = prev_date.map(index_series)
        else:
            new_data[idx_col] = np.nan
        new_columns.append(idx_col)

    if new_data:
        df = pd.concat([df, pd.DataFrame(new_data, index=df.index)], axis=1)

    return df, new_columns


def s3_create_lookback_features(df, weeks_back_list):
    df = df.sort_values(["gemrate_id", "grade", "date"]).reset_index(drop=True)
    df["week_start"] = df["date"].dt.to_period("W").dt.start_time

    agg_cols = [
        "price",
        "half_grade",
        "grade_co_BGS",
        "grade_co_CGC",
        "grade_co_PSA",
    ]

    agg_cols = [col for col in agg_cols if col in df.columns]

    weekly_agg = (
        df.groupby(["gemrate_id", "grade", "week_start"])[agg_cols].mean().reset_index()
    )

    base_rename = {c: f"avg_{c}" for c in agg_cols}
    weekly_agg = weekly_agg.rename(columns=base_rename)

    result_df = df.copy()
    new_columns = []

    for w in weeks_back_list:
        if w > 0:
            shifted = weekly_agg.copy()
            shifted["join_week"] = shifted["week_start"] + pd.Timedelta(weeks=w)

            suffix = f"_{w}w_ago"
            cols_to_add = list(base_rename.values())
            rename_dict = {c: f"{c}{suffix}" for c in cols_to_add}
            shifted = shifted.rename(columns=rename_dict)

            final_cols = list(rename_dict.values())
            new_columns.extend(final_cols)

            result_df = result_df.merge(
                shifted[["gemrate_id", "grade", "join_week"] + final_cols],
                left_on=["gemrate_id", "grade", "week_start"],
                right_on=["gemrate_id", "grade", "join_week"],
                how="left",
            ).drop(columns=["join_week"])
        else:

            suffix = f"_{w}w_ago"

            daily_agg = (
                df.groupby(["gemrate_id", "grade", "week_start", "date"])[agg_cols]
                .agg(["sum", "count"])
                .reset_index()
            )

            daily_agg.columns = ["gemrate_id", "grade", "week_start", "date"] + [
                f"{c}_{stat}" for c in agg_cols for stat in ["sum", "count"]
            ]

            daily_agg = daily_agg.sort_values(["gemrate_id", "grade", "date"])

            grp = daily_agg.groupby(["gemrate_id", "grade", "week_start"])

            for col in agg_cols:
                sum_col = f"{col}_sum"
                count_col = f"{col}_count"

                daily_agg[f"cum_{sum_col}"] = grp[sum_col].cumsum().shift(1)
                daily_agg[f"cum_{count_col}"] = grp[count_col].cumsum().shift(1)

                avg_col_name = f"avg_{col}{suffix}"
                daily_agg[avg_col_name] = (
                    daily_agg[f"cum_{sum_col}"] / daily_agg[f"cum_{count_col}"]
                )

                new_columns.append(avg_col_name)

            cols_to_merge = ["gemrate_id", "grade", "date"] + [
                f"avg_{c}{suffix}" for c in agg_cols
            ]

            result_df = result_df.merge(
                daily_agg[cols_to_merge], on=["gemrate_id", "grade", "date"], how="left"
            )

    return result_df, new_columns


def s3_create_adjacent_grade_features(df, n_sales_back):
    df = df.sort_values(["gemrate_id", "grade", "date"]).reset_index(drop=True)

    feature_cols = [
        "price",
        "half_grade",
        "grade_co_BGS",
        "grade_co_CGC",
        "grade_co_PSA",
    ]

    feature_cols = [col for col in feature_cols if col in df.columns]
    new_columns = []

    ref = df[["gemrate_id", "grade", "date"] + feature_cols].copy()
    ref["_sale_idx"] = ref.groupby(["gemrate_id", "grade"]).cumcount()

    new_data = {}

    for direction, grade_offset in [("above", 1), ("below", -1)]:
        target_grade = (df["grade"] + grade_offset).clip(lower=1.0, upper=11.0)

        df_lookup = df[["gemrate_id", "date"]].copy()
        df_lookup["target_grade"] = target_grade
        df_lookup["_orig_row"] = df.index
        df_lookup = df_lookup.sort_values("date")

        ref_lookup = ref.rename(
            columns={"grade": "target_grade", "date": "ref_date"}
        ).sort_values("ref_date")

        merged = pd.merge_asof(
            df_lookup,
            ref_lookup[["gemrate_id", "target_grade", "ref_date", "_sale_idx"]],
            left_on="date",
            right_on="ref_date",
            by=["gemrate_id", "target_grade"],
            direction="backward",
            allow_exact_matches=False,
        )

        for n in range(1, n_sales_back + 1):
            suffix = f"prev_{n}_{direction}"
            merged["_lookup_idx"] = merged["_sale_idx"] - (n - 1)

            ref_feats = ref[
                ["gemrate_id", "grade", "_sale_idx", "date"] + feature_cols
            ].copy()
            ref_feats.columns = [
                "gemrate_id",
                "target_grade",
                "_lookup_idx",
                "prev_date",
            ] + feature_cols

            with_feats = merged.merge(
                ref_feats,
                on=["gemrate_id", "target_grade", "_lookup_idx"],
                how="left",
            )
            with_feats = with_feats.set_index("_orig_row").reindex(df.index)

            for col in feature_cols:
                new_col = f"{suffix}_{col}"
                new_data[new_col] = with_feats[col]
                new_columns.append(new_col)

            new_data[f"{suffix}_days_ago"] = (
                with_feats["date"] - with_feats["prev_date"]
            ).dt.days
            new_columns.append(f"{suffix}_days_ago")

    if new_data:
        df = pd.concat([df, pd.DataFrame(new_data, index=df.index)], axis=1)

    return df, new_columns


def s3_prepare_index_data(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Skipping index data.")
        return None

    print(f"Loading and preparing index data from {filepath}...")
    idx_df = pd.read_csv(filepath)
    idx_df["date"] = pd.to_datetime(idx_df["date"])
    idx_df = idx_df.sort_values("date")

    idx_df["index_change_1d"] = idx_df["index_value"].diff()
    idx_df["index_change_1w"] = idx_df["index_value"].diff(7)
    idx_df["index_ema_12"] = (
        idx_df["index_value"].ewm(span=constants.S3_INDEX_EMA_SHORT_SPAN).mean()
    )
    idx_df["index_ema_26"] = (
        idx_df["index_value"].ewm(span=constants.S3_INDEX_EMA_LONG_SPAN).mean()
    )
    return idx_df


def s3_load_and_clean_ebay(filepath):
    """Load and clean eBay data."""
    print(f"Loading eBay data from {filepath}...")
    df = pd.read_csv(filepath)

    df["date"] = pd.to_datetime(df["item_data.date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["universal_gemrate_id"] = df["gemrate_data.universal_gemrate_id"]
    df["gemrate_id"] = df["gemrate_data.gemrate_id"]
    df.loc[df["universal_gemrate_id"].notna(), "gemrate_id"] = df.loc[
        df["universal_gemrate_id"].notna(), "universal_gemrate_id"
    ]
    df = df.dropna(subset=["gemrate_id"])
    df["grade"] = df["gemrate_data.grade"].apply(s3_clean_grade)
    df = df.dropna(subset=["grade"])
    df[["grade", "half_grade"]] = df["grade"].apply(
        lambda x: pd.Series(s3_process_grade(x))
    )

    currency_groups = (
        df["item_data.price"].astype(str).str.split().str[0].apply(s3_group_currencies)
    )
    df = df.loc[currency_groups.isin(["$ (No Country Code)", "US"])]
    df["price"] = df["item_data.price"].astype(str).apply(extract_numbers).astype(float)
    df = df[df["item_data.number_of_bids"] >= constants.S3_NUMBER_OF_BIDS_FILTER]
    df["seller"] = df["item_data.seller_name"]
    print(f"  → eBay cleaned: {len(df)} rows")
    return df


def s3_load_and_clean_pwcc(filepath):
    print(f"Loading PWCC data from {filepath}...")
    if not os.path.exists(filepath):
        print(f"  → {filepath} not found, skipping PWCC")
        return pd.DataFrame()

    df = pd.read_csv(filepath)
    if df.empty:
        return df

    df["api_response.soldDate"] = df["api_response.soldDate"].replace({None: pd.NaT})
    df = df.dropna(subset=["api_response.soldDate"])
    df["api_response.soldDate"] = (
        df["api_response.soldDate"]
        .astype(str)
        .str.replace(r" [A-Z]{3,4}$", "", regex=True)
    )
    df["date"] = pd.to_datetime(df["api_response.soldDate"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["universal_gemrate_id"] = df["gemrate_data.universal_gemrate_id"]
    df["gemrate_id"] = df["gemrate_data.gemrate_id"]
    df.loc[df["universal_gemrate_id"].notna(), "gemrate_id"] = df.loc[
        df["universal_gemrate_id"].notna(), "universal_gemrate_id"
    ]
    df = df.dropna(subset=["gemrate_id"])
    df["grade"] = df["gemrate_data.grade"].apply(s3_clean_grade)
    df = df.dropna(subset=["grade"])
    df[["grade", "half_grade"]] = df["grade"].apply(
        lambda x: pd.Series(s3_process_grade(x))
    )

    df["price"] = pd.to_numeric(df["api_response.purchasePrice"], errors="coerce")
    df = df.dropna(subset=["price"])
    df["seller"] = "fanatics"
    print(f"  → PWCC cleaned: {len(df)} rows")
    return df


def s3_load_today():
    catalog = pd.read_parquet(constants.S2_OUTPUT_EMBEDDINGS_FILE)
    df = pd.DataFrame()
    df["gemrate_id"] = catalog["gemrate_id"]
    all_today = []
    for grading_company in ["PSA", "CGC", "BGS"]:
        for grade in range(constants.S3_LOWEST_GRADE, constants.S3_HIGHEST_GRADE + 1):
            for half_grade in [0, 1]:
                if grade == 11 and half_grade == 1:
                    continue
                elif grading_company != "BGS" and grade == 11:
                    continue
                elif grading_company == "PSA" and grade == 10 and half_grade == 1:
                    continue
                else:
                    df = pd.DataFrame()
                    df["gemrate_id"] = catalog["gemrate_id"]
                    df["grade"] = grade
                    df["half_grade"] = half_grade
                    df["grading_company"] = grading_company
                    df["seller_popularity"] = 0.3
                    all_today.append(df)
    df = pd.concat(all_today)
    df["date"] = pd.to_datetime("today")
    df = df.reset_index(drop=True)
    return df


def run_step_3():
    print("Starting Step 3: Feature Prep...")
    start_time = time.time()

    ebay_df = s3_load_and_clean_ebay(constants.S1_EBAY_MARKET_FILE)
    pwcc_df = s3_load_and_clean_pwcc(constants.S1_PWCC_MARKET_FILE)
    today = s3_load_today()

    common_cols = [
        "gemrate_id",
        "date",
        "price",
        "grade",
        "half_grade",
        "grading_company",
        "seller",
    ]

    ebay_subset = ebay_df[[c for c in common_cols if c in ebay_df.columns]].copy()
    pwcc_subset = pwcc_df[[c for c in common_cols if c in pwcc_df.columns]].copy()
    ebay_subset.to_parquet("ebay_cleaned.parquet")
    pwcc_subset.to_parquet("fanatics_cleaned.parquet")
    print("Merging eBay and PWCC data...")
    df = pd.concat([ebay_subset, pwcc_subset], ignore_index=True)
    df = df.loc[df["grade"] >= constants.S3_LOWEST_GRADE]
    df = df.loc[df["grade"] <= constants.S3_HIGHEST_GRADE]
    df["price"] = df["price"].clip(lower=0.01)
    df = df.reset_index(drop=True)
    df = s3_calculate_seller_popularity(df)

    # Calculate sales per day for data integrity tracking (last 28 days)
    # cutoff_date = pd.Timestamp.now().normalize() - pd.Timedelta(days=28)

    cutoff_date = pd.to_datetime("2025-12-1") - pd.Timedelta(days=28)
    recent_sales = df[df["date"] >= cutoff_date]
    sales_per_day = recent_sales.groupby(recent_sales["date"].dt.date).size()
    sales_per_day = sales_per_day.sort_index()

    # Add sales per day chart to data integrity tracker
    tracker = get_tracker()

    chart_data = [[str(d), int(count)] for d, count in sales_per_day.items()]
    tracker.add_chart(
        id="sales_per_day",
        title="Sales Per Day",
        chart_type="line",
        columns=["date", "sales"],
        data=chart_data,
    )
    print(
        f"  → Recorded {len(sales_per_day)} days of sales data for integrity tracking"
    )

    # Calculate median days between sales for top 100 cards by sales volume
    print("  → Calculating median days between sales for top 100 cards...")
    top_100_ids = df["gemrate_id"].value_counts().head(100).index.tolist()
    top_100_df = df[df["gemrate_id"].isin(top_100_ids)].copy()
    top_100_df = top_100_df.sort_values(["gemrate_id", "date"]).reset_index(drop=True)
    top_100_df["prev_sale_date"] = top_100_df.groupby("gemrate_id")["date"].shift(1)
    top_100_df["days_between"] = (
        top_100_df["date"] - top_100_df["prev_sale_date"]
    ).dt.days
    days_between_all = top_100_df["days_between"].dropna()
    if len(days_between_all) > 0:
        median_days = days_between_all.median()
        tracker.add_metric(
            id="s3_median_days_between_top_100_sales",
            title="Median Days Between Sales (Top 100 Cards)",
            value=f"{median_days:.1f} days",
        )
        print(f"  → Median days between sales for top 100 cards: {median_days:.1f}")

    # Calculate sales concentration per day (total sales vs unique cards)
    print("  → Calculating sales concentration per day...")
    concentration_data = []
    for sale_date in sorted(sales_per_day.index):
        day_sales = recent_sales[recent_sales["date"].dt.date == sale_date]
        total_sales = len(day_sales)
        unique_cards = day_sales["gemrate_id"].nunique()
        concentration_data.append([str(sale_date), total_sales, unique_cards])

    tracker.add_table(
        id="sales_concentration_per_day",
        title="Sales Concentration Per Day",
        columns=["date", "total_sales", "unique_cards"],
        data=concentration_data,
    )
    print(f"  → Recorded {len(concentration_data)} days of sales concentration data")

    # df = df.loc[df["seller_popularity"] >= 0.01]
    df = pd.concat([df, today], ignore_index=True)
    print(f"  → Total merged: {len(df)} rows")
    dummies = pd.get_dummies(df["grading_company"], prefix="grade_co", dtype=int)
    df = pd.concat([df, dummies], axis=1)

    columns_to_keep = [
        "gemrate_id",
        "date",
        "price",
        "grade",
        "half_grade",
        "seller_popularity",
    ]

    columns_to_keep.extend(dummies.columns.tolist())
    df = df[columns_to_keep].copy()

    catalog_df = pd.read_csv(constants.S2_INPUT_CATALOG_FILE, low_memory=False)
    mask_univ = catalog_df["UNIVERSAL_GEMRATE_ID"].notna()
    catalog_df.loc[mask_univ, "GEMRATE_ID"] = catalog_df.loc[
        mask_univ, "UNIVERSAL_GEMRATE_ID"
    ]
    catalog_df = catalog_df.drop_duplicates(subset=["GEMRATE_ID"])

    valid_ids = set(catalog_df["GEMRATE_ID"].astype(str))
    initial_count = len(df)

    df = df[df["gemrate_id"].astype(str).isin(valid_ids)]
    print(
        f"  → Filtered {initial_count - len(df)} rows not in catalog. Remaining: {len(df)}"
    )

    tracker.add_metric(
        id="s3_total_cleaned",
        title="Total Filtered Records",
        value=f"{len(df):,}",
    )

    unique_ids = df["gemrate_id"].unique()
    batch_size = constants.S3_BATCH_SIZE

    index_df = s3_prepare_index_data(constants.S1_INDEX_FILE)

    n_batches = int(np.ceil(len(unique_ids) / batch_size))
    print(f"Processing in {n_batches} batches of {batch_size} IDs...")

    today_date = pd.to_datetime("today").normalize()

    hist_writer = None
    today_writer = None
    first_time_sales_by_date = {}  # Track first-time sales per day

    import pyarrow as pa
    import pyarrow.parquet as pq

    for i in tqdm(range(0, len(unique_ids), batch_size), desc="Processing Batches"):
        batch_ids = unique_ids[i : i + batch_size]
        batch_df = df[df["gemrate_id"].isin(batch_ids)].copy()

        batch_df, _ = s3_create_previous_sale_features(
            batch_df, constants.S3_N_SALES_BACK
        )

        # Track first-time sales (cards with no previous sales) in last 28 days
        if "prev_1_days_ago" in batch_df.columns:
            first_time_mask = (
                batch_df["prev_1_days_ago"].isna()
                & batch_df["price"].notna()
                & (batch_df["date"] >= cutoff_date)
            )
            first_time_df = batch_df[first_time_mask]
            for sale_date in first_time_df["date"].dt.date:
                first_time_sales_by_date[sale_date] = (
                    first_time_sales_by_date.get(sale_date, 0) + 1
                )

        batch_df, _ = s3_create_lookback_features(
            batch_df, constants.S3_WEEKS_BACK_LIST
        )
        batch_df, _ = s3_create_adjacent_grade_features(
            batch_df, constants.S3_N_SALES_BACK
        )

        if index_df is not None:
            batch_df = batch_df.merge(index_df, on="date", how="left")

        if "week_start" in batch_df.columns:
            batch_df = batch_df.drop("week_start", axis=1)

        batch_df["date"] = pd.to_datetime(batch_df["date"])

        today_mask = (batch_df["date"].dt.normalize() == today_date) & (
            batch_df["price"].isna()
        )
        df_today = batch_df[today_mask]
        df_hist = batch_df[~today_mask]

        if not df_hist.empty:
            table_hist = pa.Table.from_pandas(df_hist, preserve_index=False)
            if hist_writer is None:
                hist_writer = pq.ParquetWriter(
                    constants.S3_HISTORICAL_DATA_FILE, table_hist.schema
                )
            hist_writer.write_table(table_hist)

        if not df_today.empty:
            table_today = pa.Table.from_pandas(df_today, preserve_index=False)
            if today_writer is None:
                today_writer = pq.ParquetWriter(
                    constants.S3_TODAY_DATA_FILE, table_today.schema
                )
            today_writer.write_table(table_today)

        del batch_df, df_hist, df_today
        import gc

        gc.collect()

    if hist_writer:
        hist_writer.close()
    if today_writer:
        today_writer.close()

    print(f"Saved historical data to {constants.S3_HISTORICAL_DATA_FILE}")
    print(f"Saved today's data to {constants.S3_TODAY_DATA_FILE}")

    # Add first-time sales per day chart to data integrity tracker
    if first_time_sales_by_date:
        sorted_dates = sorted(first_time_sales_by_date.keys())
        first_time_chart_data = [
            [str(d), int(first_time_sales_by_date[d])] for d in sorted_dates
        ]
        tracker.add_chart(
            id="first_time_sales_per_day",
            title="First-Time Sales Per Day (No Previous Sales)",
            chart_type="line",
            columns=["date", "first_time_sales"],
            data=first_time_chart_data,
        )
        print(
            f"  → Recorded {len(first_time_sales_by_date)} days of first-time sales data"
        )

    # Data Integrity Tracking - Duration
    duration = time.time() - start_time
    tracker.add_metric(
        id="s3_duration",
        title="Step 3 Duration",
        value=f"{duration:.1f}s",
    )

    print("Step 3 Complete.")
