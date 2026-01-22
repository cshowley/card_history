from datetime import datetime, timedelta
import time
import pandas as pd
import requests
from pymongo import MongoClient
import constants
from data_integrity import get_tracker


def s1_fetch_index_data(url):
    response = requests.get(url)
    response.raise_for_status()
    index_data = response.json()
    index_df = pd.DataFrame(index_data)
    index_df["date"] = pd.to_datetime(index_df["date"])
    index_df = index_df.rename(columns={"value": "index_value"})
    print(f"Downloaded {len(index_df)} index data points")
    print(f"Index date range: {index_df['date'].min()} to {index_df['date'].max()}")
    return index_df


def run_step_1():
    print("Starting Step 1: Download Sales...")
    start_time = time.time()
    tracker = get_tracker()

    if not constants.S1_MONGO_URL:
        tracker.add_error("MONGO_URL environment variable is not set", step="step_1")
        raise ValueError("MONGO_URL environment variable is not set")
    client = MongoClient(constants.S1_MONGO_URL)
    db = client[constants.S1_DB_NAME]

    #################ADD TRAINING CUTOFF START DATE
    # df = df[df.date >= datetime(2025, 9, 1)]
    ########something like this ^^^^^^
    ######## currently in step 7
    ########### or modify the queries using filters that correctly catch string-ified dates

    ##########
    # Calculate lower and upper date bounds for string prefix comparison (works for "YYYY-MM-DDTHH:MM:SS..." formats)
    days_back = 7  # Adjustable: days for lower bound
    upper_cutoff_date = datetime(2025, 12, 1)  # datetime.now()
    lower_cutoff_date = upper_cutoff_date - timedelta(days=days_back)
    lower_date_str = lower_cutoff_date.strftime("%Y-%m-%d")
    upper_bound_str = (upper_cutoff_date + timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )  # Exclusive upper for full day inclusion
    print(f"Date range filter: >= '{lower_date_str}' and < '{upper_bound_str}'")
    ##########

    print("Downloading eBay sales...")
    ebay_collection = db[constants.S1_EBAY_COLLECTION]
    ebay_pipeline = [
        {
            "$match": {
                "$or": [
                    {"gemrate_data.universal_gemrate_id": {"$exists": True, "$ne": ""}},
                    {"gemrate_data.gemrate_id": {"$exists": True, "$ne": ""}},
                ],
                "item_data.date": {
                    "$exists": True,
                    "$gte": lower_date_str,
                    "$lt": upper_bound_str,
                },
                "item_data.price": {"$exists": True},
                "item_data.format": "auction",
            }
        },
        {
            "$project": {
                "gemrate_data.universal_gemrate_id": 1,
                "gemrate_data.gemrate_id": 1,
                "item_data.date": 1,
                "grading_company": 1,
                "gemrate_data.grade": 1,
                "item_data.price": 1,
                "item_data.number_of_bids": 1,
                "item_data.seller_name": 1,
                "_id": 1,
            }
        },
    ]
    ebay_results = ebay_collection.aggregate(
        ebay_pipeline, maxTimeMS=constants.S1_MONGO_MAX_TIME_MS, allowDiskUse=True
    )
    ebay_df = pd.json_normalize(list(ebay_results))
    print(f" → eBay rows loaded: {len(ebay_df)}")
    ebay_df.to_csv(constants.S1_EBAY_MARKET_FILE, index=False)

    print("Downloading PWCC/Fanatics sales...")
    pwcc_collection = db[constants.S1_PWCC_COLLECTION]
    pwcc_pipeline = [
        {
            "$match": {
                "$or": [
                    {"gemrate_data.universal_gemrate_id": {"$exists": True, "$ne": ""}},
                    {"gemrate_data.gemrate_id": {"$exists": True, "$ne": ""}},
                ],
                "api_response.soldDate": {
                    "$exists": True,
                    "$gte": lower_date_str,
                    "$lt": upper_bound_str,
                },
                "api_response.purchasePrice": {"$exists": True},
                "api_response.auctionType": {"$in": ["WEEKLY", "PREMIUM"]},
            }
        },
        {
            "$project": {
                "gemrate_data.universal_gemrate_id": 1,
                "gemrate_data.gemrate_id": 1,
                "api_response.soldDate": 1,
                "api_response.purchasePrice": 1,
                "api_response.auctionType": 1,
                "api_response.gradingService": 1,
                "gemrate_data.grade": 1,
                "_id": 1,
            }
        },
    ]
    pwcc_results = pwcc_collection.aggregate(
        pwcc_pipeline, maxTimeMS=constants.S1_MONGO_MAX_TIME_MS, allowDiskUse=True
    )
    pwcc_df = pd.json_normalize(list(pwcc_results))
    print(f" → PWCC rows loaded: {len(pwcc_df)}")
    pwcc_df.to_csv(constants.S1_PWCC_MARKET_FILE, index=False)

    index_df = s1_fetch_index_data(constants.S1_INDEX_API_URL)
    index_df.to_csv(constants.S1_INDEX_FILE, index=False)

    # Data Integrity Tracking
    duration = time.time() - start_time
    ebay_count = len(ebay_df)
    pwcc_count = len(pwcc_df)
    total_count = ebay_count + pwcc_count

    # Count sales before 9/1/2025
    cutoff_date = datetime(2025, 9, 1)
    ebay_before_cutoff = 0
    pwcc_before_cutoff = 0

    if "item_data.date" in ebay_df.columns and len(ebay_df) > 0:
        ebay_dates = pd.to_datetime(ebay_df["item_data.date"], errors="coerce")
        ebay_before_cutoff = (ebay_dates < cutoff_date).sum()

    if "api_response.soldDate" in pwcc_df.columns and len(pwcc_df) > 0:
        # Remove timezone abbreviation from the end of the sold date
        pwcc_dates_str = pwcc_df["api_response.soldDate"].str.replace(
            r" [A-Z]{3,4}$", "", regex=True
        )
        pwcc_dates = pd.to_datetime(pwcc_dates_str, errors="coerce")
        pwcc_before_cutoff = (pwcc_dates < cutoff_date).sum()

    sales_before_cutoff = ebay_before_cutoff + pwcc_before_cutoff
    print(f"sales before cutoff: {sales_before_cutoff}")
    tracker.add_metric(
        id="s1_sales_before_sep_2025",
        title="Sales Before 9/1/2025",
        value=f"{sales_before_cutoff:,}",
    )

    tracker.add_metric(
        id="s1_total_records",
        title="Total Records Downloaded",
        value=f"{total_count:,}",
    )
    tracker.add_metric(
        id="s1_duration",
        title="Step 1 Duration",
        value=f"{duration:.1f}s",
    )
    tracker.add_table(
        id="s1_marketplace_breakdown",
        title="Marketplace Breakdown",
        columns=["Source", "Records", "Share"],
        data=[
            [
                "eBay",
                f"{ebay_count:,}",
                f"{ebay_count/total_count*100:.1f}%" if total_count > 0 else "0%",
            ],
            [
                "PWCC",
                f"{pwcc_count:,}",
                f"{pwcc_count/total_count*100:.1f}%" if total_count > 0 else "0%",
            ],
        ],
    )

    print("Step 1 Complete.")

    # # ==========================================================
    # # Data Integrity Tracking
    # # ==========================================================
    # duration = time.time() - start_time
    # ebay_count = len(ebay_df)
    # pwcc_count = len(pwcc_df)
    # total_count = ebay_count + pwcc_count

    # # Volume metrics
    # tracker.add_metric("s1_ebay_records", "eBay Records Downloaded", f"{ebay_count:,}")
    # tracker.add_metric("s1_pwcc_records", "PWCC Records Downloaded", f"{pwcc_count:,}")
    # tracker.add_metric(
    #     "s1_total_records", "Total Records Downloaded", f"{total_count:,}"
    # )
    # tracker.add_metric("s1_duration", "Step 1 Duration", f"{duration:.1f}s")

    # # Market share
    # if total_count > 0:
    #     ebay_share = ebay_count / total_count * 100
    #     tracker.add_metric("s1_ebay_share", "eBay Market Share", f"{ebay_share:.1f}%")

    # # Missing data analysis - eBay
    # ebay_missing = []
    # if "gemrate_data.gemrate_id" in ebay_df.columns:
    #     no_gid = (
    #         ebay_df["gemrate_data.gemrate_id"].isna().sum()
    #         + (ebay_df["gemrate_data.gemrate_id"] == "").sum()
    #     )
    #     ebay_missing.append(
    #         [
    #             "gemrate_id",
    #             f"{no_gid:,}",
    #             f"{no_gid/ebay_count*100:.2f}%" if ebay_count > 0 else "0%",
    #         ]
    #     )
    # if "gemrate_data.grade" in ebay_df.columns:
    #     no_grade = (
    #         ebay_df["gemrate_data.grade"].isna().sum()
    #         + (ebay_df["gemrate_data.grade"] == "").sum()
    #     )
    #     ebay_missing.append(
    #         [
    #             "grade",
    #             f"{no_grade:,}",
    #             f"{no_grade/ebay_count*100:.2f}%" if ebay_count > 0 else "0%",
    #         ]
    #     )
    # if "item_data.price" in ebay_df.columns:
    #     no_price = ebay_df["item_data.price"].isna().sum()
    #     ebay_missing.append(
    #         [
    #             "price",
    #             f"{no_price:,}",
    #             f"{no_price/ebay_count*100:.2f}%" if ebay_count > 0 else "0%",
    #         ]
    #     )

    # if ebay_missing:
    #     tracker.add_table(
    #         "s1_ebay_missing",
    #         "eBay Missing Data",
    #         ["Field", "Count", "Pct"],
    #         ebay_missing,
    #         col_span=6,
    #     )

    # # Missing data analysis - PWCC
    # pwcc_missing = []
    # if "gemrate_data.gemrate_id" in pwcc_df.columns:
    #     no_gid = (
    #         pwcc_df["gemrate_data.gemrate_id"].isna().sum()
    #         + (pwcc_df["gemrate_data.gemrate_id"] == "").sum()
    #     )
    #     pwcc_missing.append(
    #         [
    #             "gemrate_id",
    #             f"{no_gid:,}",
    #             f"{no_gid/pwcc_count*100:.2f}%" if pwcc_count > 0 else "0%",
    #         ]
    #     )
    # if "gemrate_data.grade" in pwcc_df.columns:
    #     no_grade = (
    #         pwcc_df["gemrate_data.grade"].isna().sum()
    #         + (pwcc_df["gemrate_data.grade"] == "").sum()
    #     )
    #     pwcc_missing.append(
    #         [
    #             "grade",
    #             f"{no_grade:,}",
    #             f"{no_grade/pwcc_count*100:.2f}%" if pwcc_count > 0 else "0%",
    #         ]
    #     )
    # if "api_response.purchasePrice" in pwcc_df.columns:
    #     no_price = pwcc_df["api_response.purchasePrice"].isna().sum()
    #     pwcc_missing.append(
    #         [
    #             "price",
    #             f"{no_price:,}",
    #             f"{no_price/pwcc_count*100:.2f}%" if pwcc_count > 0 else "0%",
    #         ]
    #     )

    # if pwcc_missing:
    #     tracker.add_table(
    #         "s1_pwcc_missing",
    #         "PWCC Missing Data",
    #         ["Field", "Count", "Pct"],
    #         pwcc_missing,
    #         col_span=6,
    #     )

    # # Grade distribution - eBay
    # if "gemrate_data.grade" in ebay_df.columns and ebay_count > 0:
    #     grade_counts = (
    #         ebay_df["gemrate_data.grade"].fillna("None").value_counts().head(10)
    #     )
    #     tracker.add_chart(
    #         "s1_ebay_grades",
    #         "eBay Grade Distribution",
    #         "bar",
    #         grade_counts.index.tolist(),
    #         [{"label": "Count", "data": grade_counts.values.tolist()}],
    #         col_span=6,
    #     )

    # # Grade distribution - PWCC
    # if "gemrate_data.grade" in pwcc_df.columns and pwcc_count > 0:
    #     grade_counts = (
    #         pwcc_df["gemrate_data.grade"].fillna("None").value_counts().head(10)
    #     )
    #     tracker.add_chart(
    #         "s1_pwcc_grades",
    #         "PWCC Grade Distribution",
    #         "bar",
    #         grade_counts.index.tolist(),
    #         [{"label": "Count", "data": grade_counts.values.tolist()}],
    #         col_span=6,
    #     )

    # # Grading company distribution - eBay
    # if "grading_company" in ebay_df.columns and ebay_count > 0:
    #     company_counts = (
    #         ebay_df["grading_company"].fillna("None").value_counts().head(10)
    #     )
    #     tracker.add_chart(
    #         "s1_grading_companies",
    #         "Grading Company Distribution",
    #         "pie",
    #         company_counts.index.tolist(),
    #         [{"label": "Count", "data": company_counts.values.tolist()}],
    #         col_span=6,
    #     )

    # # Auction type distribution - PWCC
    # if "api_response.auctionType" in pwcc_df.columns and pwcc_count > 0:
    #     auction_counts = (
    #         pwcc_df["api_response.auctionType"].fillna("Unknown").value_counts()
    #     )
    #     tracker.add_chart(
    #         "s1_auction_types",
    #         "PWCC Auction Types",
    #         "pie",
    #         auction_counts.index.tolist(),
    #         [{"label": "Count", "data": auction_counts.values.tolist()}],
    #         col_span=6,
    #     )

    # # Price statistics - eBay
    # if "item_data.price" in ebay_df.columns and ebay_count > 0:
    #     prices = ebay_df["item_data.price"].copy()
    #     if prices.dtype == object:
    #         prices = prices.str.replace(r"[\$,]", "", regex=True)
    #         prices = pd.to_numeric(prices, errors="coerce")
    #     prices = prices.dropna()
    #     if len(prices) > 0:
    #         price_rows = [
    #             ["Average", f"${prices.mean():,.2f}"],
    #             ["Median", f"${prices.median():,.2f}"],
    #             ["Min", f"${prices.min():,.2f}"],
    #             ["Max", f"${prices.max():,.2f}"],
    #         ]
    #         tracker.add_table(
    #             "s1_ebay_prices",
    #             "eBay Price Stats",
    #             ["Metric", "Value"],
    #             price_rows,
    #             col_span=6,
    #         )

    # # Price statistics - PWCC
    # if "api_response.purchasePrice" in pwcc_df.columns and pwcc_count > 0:
    #     prices = pd.to_numeric(
    #         pwcc_df["api_response.purchasePrice"], errors="coerce"
    #     ).dropna()
    #     if len(prices) > 0:
    #         price_rows = [
    #             ["Average", f"${prices.mean():,.2f}"],
    #             ["Median", f"${prices.median():,.2f}"],
    #             ["Min", f"${prices.min():,.2f}"],
    #             ["Max", f"${prices.max():,.2f}"],
    #         ]
    #         tracker.add_table(
    #             "s1_pwcc_prices",
    #             "PWCC Price Stats",
    #             ["Metric", "Value"],
    #             price_rows,
    #             col_span=6,
    #         )

    # # Anomaly detection - eBay bid counts
    # if "item_data.number_of_bids" in ebay_df.columns and ebay_count > 0:
    #     bids = pd.to_numeric(ebay_df["item_data.number_of_bids"], errors="coerce")
    #     zero_bid = (bids == 0).sum()
    #     single_bid = (bids == 1).sum()
    #     anomaly_rows = [
    #         ["Zero-bid auctions", f"{zero_bid:,}"],
    #         ["Single-bid auctions", f"{single_bid:,}"],
    #     ]
    #     tracker.add_table(
    #         "s1_anomalies",
    #         "Data Anomalies",
    #         ["Type", "Count"],
    #         anomaly_rows,
    #         col_span=6,
    #     )

    # # Daily sales trend - eBay
    # if "item_data.date" in ebay_df.columns and ebay_count > 0:
    #     dates = pd.to_datetime(ebay_df["item_data.date"], errors="coerce").dt.date
    #     daily_counts = dates.value_counts().sort_index().tail(10)
    #     tracker.add_chart(
    #         "s1_daily_sales",
    #         "Daily eBay Sales",
    #         "line",
    #         [str(d) for d in daily_counts.index.tolist()],
    #         [{"label": "Sales", "data": daily_counts.values.tolist()}],
    #         col_span=12,
    #     )

    # print("Step 1 Complete.")
