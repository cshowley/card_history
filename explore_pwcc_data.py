"""
Exploratory script to analyze and integrate pwcc_graded_items (Fanatics) data
with existing ebay_graded_items data for XGBoost model training.

This script:
1. Queries both MongoDB collections and compares schemas
2. Normalizes the data into a unified format
3. Adds source one-hot encoding (ebay vs fanatics)
4. Groups features by universal_gemrate_id (not by source)
"""

import os
import re
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from pprint import pprint

load_dotenv()

# ==============================================================================
# CONFIGURATION
# ==============================================================================

MONGO_URL = os.getenv("MONGO_URL") or os.getenv("MONGO_URI")
DB_NAME = "gemrate"
EBAY_COLLECTION = "ebay_graded_items"
PWCC_COLLECTION = "pwcc_graded_items"

# ==============================================================================
# SCHEMA EXPLORATION
# ==============================================================================


def get_sample_document(collection, collection_name: str):
    """Get a sample document and print its structure."""
    print(f"\n{'='*60}")
    print(f"Sample document from {collection_name}")
    print(f"{'='*60}")

    doc = collection.find_one()
    if doc:
        pprint(doc, depth=3)
    else:
        print("No documents found!")
    return doc


def get_collection_stats(collection, collection_name: str):
    """Get basic stats about a collection."""
    print(f"\n{'='*60}")
    print(f"Collection Stats: {collection_name}")
    print(f"{'='*60}")

    count = collection.count_documents({})
    print(f"Total documents: {count:,}")

    # Check for key fields
    fields_to_check = [
        "gemrate_data.universal_gemrate_id",
        "gemrate_hybrid_data.specid",
        "item_data.date",
        "item_data.price",
        "item_data.number_of_bids",
        "item_data.format",
        "grading_company",
        "gemrate_data.grade",
    ]

    print("\nField availability:")
    for field in fields_to_check:
        has_field = collection.count_documents({field: {"$exists": True}})
        pct = (has_field / count * 100) if count > 0 else 0
        print(f"  {field}: {has_field:,} ({pct:.1f}%)")


def compare_schemas(db):
    """Compare schemas between ebay and pwcc collections."""
    ebay = db[EBAY_COLLECTION]
    pwcc = db[PWCC_COLLECTION]

    # Get sample documents
    ebay_doc = get_sample_document(ebay, EBAY_COLLECTION)
    pwcc_doc = get_sample_document(pwcc, PWCC_COLLECTION)

    # Get stats
    get_collection_stats(ebay, EBAY_COLLECTION)
    get_collection_stats(pwcc, PWCC_COLLECTION)

    return ebay_doc, pwcc_doc


# ==============================================================================
# DATA LOADING
# ==============================================================================


def load_ebay_data(collection, limit: int = None):
    """Load data from ebay_graded_items collection."""
    print(f"\nLoading data from ebay_graded_items...")

    pipeline = [
        {
            "$match": {
                "gemrate_data.universal_gemrate_id": {"$exists": True, "$ne": ""},
                "item_data.date": {"$exists": True},
                "item_data.price": {"$exists": True},
            }
        },
        {
            "$project": {
                "gemrate_data.universal_gemrate_id": 1,
                "gemrate_hybrid_data.specid": 1,
                "item_data.date": 1,
                "item_data.price": 1,
                "item_data.number_of_bids": 1,
                "item_data.format": 1,
                "item_data.seller_name": 1,
                "grading_company": 1,
                "gemrate_data.grade": 1,
                "_id": 1,
            }
        },
    ]

    if limit:
        pipeline.append({"$limit": limit})

    results = list(collection.aggregate(pipeline, maxTimeMS=600000, allowDiskUse=True))
    df = pd.json_normalize(results)

    print(f"  → Loaded {len(df)} rows")
    print(f"  → Columns: {list(df.columns)}")

    return df


def load_pwcc_data(collection, limit: int = None):
    """Load data from pwcc_graded_items collection (Fanatics)."""
    print(f"\nLoading data from pwcc_graded_items...")

    pipeline = [
        {
            "$match": {
                "gemrate_data.universal_gemrate_id": {"$exists": True, "$ne": ""},
                "api_response.soldDate": {"$exists": True},
                "api_response.purchasePrice": {"$exists": True},
            }
        },
        {
            "$project": {
                "gemrate_data.universal_gemrate_id": 1,
                "gemrate_data.specid": 1,
                "api_response.soldDate": 1,
                "api_response.purchasePrice": 1,
                "api_response.auctionType": 1,
                "api_response.gradingService": 1,
                "gemrate_data.grade": 1,
                "_id": 1,
            }
        },
    ]

    if limit:
        pipeline.append({"$limit": limit})

    results = list(collection.aggregate(pipeline, maxTimeMS=600000, allowDiskUse=True))
    df = pd.json_normalize(results)

    print(f"  → Loaded {len(df)} rows")
    print(f"  → Columns: {list(df.columns)}")

    return df


# ==============================================================================
# DATA CLEANING (same as main.py)
# ==============================================================================


def clean_grade(val):
    """Clean grade values to numeric."""
    s = str(val).lower().strip().replace("g", "").replace("_", ".")
    if s in ["nan", "none", "", "0", "auth"]:
        return np.nan

    if "10b" in s or "10black" in s:
        return 11.0

    if any(x in s for x in ["pristine", "perfect", "10p"]):
        return 10.5

    match = re.search(r"(\d+(\.\d+)?)", s)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return np.nan
    return np.nan


def process_grade(val):
    """Split grade into floor and half components."""
    if pd.isna(val):
        return np.nan, np.nan
    floor_val = np.floor(val)
    half_val = 1.0 if (val - floor_val) > 0 else 0.0
    return floor_val, half_val


def group_currencies(val):
    """Normalize currency strings."""
    s = str(val).strip()
    if not s:
        return "Unknown"
    if s.startswith("$") or s[0].isdigit():
        return "$ (No Country Code)"
    return s


def clean_ebay_dataframe(df):
    """Clean and standardize eBay data."""
    print(f"\nCleaning ebay data...")

    # Standardize column names
    df = df.rename(
        columns={
            "gemrate_data.universal_gemrate_id": "universal_gemrate_id",
            "gemrate_hybrid_data.specid": "spec_id",
            "item_data.date": "date_raw",
            "item_data.price": "price_raw",
            "item_data.number_of_bids": "number_of_bids",
            "item_data.format": "format",
            "item_data.seller_name": "seller_name",
            "gemrate_data.grade": "grade_raw",
        }
    )

    # Date conversion
    df["date"] = pd.to_datetime(df["date_raw"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Universal ID
    df["universal_gemrate_id"] = df["universal_gemrate_id"].astype(str)
    df = df.dropna(subset=["universal_gemrate_id"])

    # Grade cleaning
    df["grade"] = df["grade_raw"].apply(clean_grade)
    df = df.dropna(subset=["grade"])
    df[["grade", "half_grade"]] = df["grade"].apply(
        lambda x: pd.Series(process_grade(x))
    )

    # Price cleaning
    df["price_str"] = df["price_raw"].astype(str)
    currency_groups = df["price_str"].str.split().str[0].apply(group_currencies)
    df = df.loc[currency_groups.isin(["$ (No Country Code)", "US"])]

    df["price"] = df["price_str"].str.replace(r"\D+", "", regex=True).astype(float)
    df["price"] = np.log(df["price"].clip(lower=0.01))

    # Number of bids
    df["number_of_bids"] = df["number_of_bids"].fillna(0).astype(int)

    # Grading company dummies
    df["grading_company"] = df["grading_company"].fillna("Unknown")
    dummies = pd.get_dummies(df["grading_company"], prefix="grade_co", dtype=int)
    df = pd.concat([df, dummies], axis=1)

    # Add source indicator
    df["source"] = "ebay"

    print(f"  → After cleaning: {len(df)} rows")

    return df


def clean_pwcc_dataframe(df):
    """Clean and standardize PWCC/Fanatics data."""
    print(f"\nCleaning fanatics data...")

    if df.empty:
        print("  → No data to clean")
        return df

    # Standardize column names (different from eBay!)
    df = df.rename(
        columns={
            "gemrate_data.universal_gemrate_id": "universal_gemrate_id",
            "gemrate_data.specid": "spec_id",
            "api_response.soldDate": "date_raw",
            "api_response.purchasePrice": "price_raw",
            "api_response.auctionType": "format",
            "api_response.gradingService": "grading_company",
            "gemrate_data.grade": "grade_raw",
        }
    )

    # Date conversion (PWCC format: "2025-10-01T11:50:49.000 PDT")
    # Replace Python None with NaT, then drop
    df["date_raw"] = df["date_raw"].replace({None: pd.NaT})
    df = df.dropna(subset=["date_raw"])
    # Strip timezone abbreviation (PDT, PST, etc.) before parsing
    df["date_raw"] = (
        df["date_raw"].astype(str).str.replace(r" [A-Z]{3,4}$", "", regex=True)
    )
    df["date"] = pd.to_datetime(df["date_raw"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Universal ID
    df["universal_gemrate_id"] = df["universal_gemrate_id"].astype(str)
    df = df.dropna(subset=["universal_gemrate_id"])

    # Grade cleaning
    df["grade"] = df["grade_raw"].apply(clean_grade)
    df = df.dropna(subset=["grade"])
    df[["grade", "half_grade"]] = df["grade"].apply(
        lambda x: pd.Series(process_grade(x))
    )

    # Price cleaning (PWCC is already numeric USD)
    df["price"] = pd.to_numeric(df["price_raw"], errors="coerce")
    df = df.dropna(subset=["price"])
    df["price"] = np.log(df["price"].clip(lower=0.01))

    # Number of bids (PWCC doesn't have this - set to 0)
    df["number_of_bids"] = 0

    # Seller name (PWCC doesn't have this - set to "fanatics")
    df["seller_name"] = "fanatics"

    # Grading company dummies
    df["grading_company"] = df["grading_company"].fillna("Unknown")
    dummies = pd.get_dummies(df["grading_company"], prefix="grade_co", dtype=int)
    df = pd.concat([df, dummies], axis=1)

    # Add source indicator
    df["source"] = "fanatics"

    print(f"  → After cleaning: {len(df)} rows")

    return df


# ==============================================================================
# MERGE AND FEATURE ENGINEERING
# ==============================================================================


def merge_sources(ebay_df: pd.DataFrame, pwcc_df: pd.DataFrame) -> pd.DataFrame:
    """Merge both sources and add one-hot source encoding."""
    print("\nMerging data sources...")

    # Ensure both have same columns for merge
    common_cols = [
        "universal_gemrate_id",
        "spec_id",
        "date",
        "price",
        "grade",
        "half_grade",
        "number_of_bids",
        "grading_company",
        "seller_name",
        "source",
    ]

    # Add grading company dummies from both
    grade_co_cols = [c for c in ebay_df.columns if c.startswith("grade_co_")]
    grade_co_cols_pwcc = [c for c in pwcc_df.columns if c.startswith("grade_co_")]
    all_grade_co_cols = list(set(grade_co_cols + grade_co_cols_pwcc))

    # Ensure both dfs have all grade_co columns
    for col in all_grade_co_cols:
        if col not in ebay_df.columns:
            ebay_df[col] = 0
        if col not in pwcc_df.columns:
            pwcc_df[col] = 0

    cols_to_use = common_cols + all_grade_co_cols

    ebay_subset = ebay_df[[c for c in cols_to_use if c in ebay_df.columns]].copy()
    pwcc_subset = pwcc_df[[c for c in cols_to_use if c in pwcc_df.columns]].copy()

    # Concatenate
    merged = pd.concat([ebay_subset, pwcc_subset], ignore_index=True)

    # Add source one-hot encoding
    merged["source_ebay"] = (merged["source"] == "ebay").astype(int)
    merged["source_fanatics"] = (merged["source"] == "fanatics").astype(int)

    print(f"  → Total merged rows: {len(merged)}")
    print(f"  → Source distribution:")
    print(merged["source"].value_counts())

    return merged


def calculate_seller_popularity(df):
    """Calculate seller popularity using expanding window (across all sources)."""
    print("\nCalculating seller popularity...")
    df = df.sort_values("date").copy()
    df["seller_cum_count"] = df.groupby("seller_name").cumcount() + 1
    df["global_cum_count"] = np.arange(1, len(df) + 1)
    df["seller_popularity"] = df["seller_cum_count"] / df["global_cum_count"]
    df.drop(columns=["seller_cum_count", "global_cum_count"], inplace=True)
    return df


def create_previous_sale_features(df, n_sales_back: int = 5):
    """Generate lag features grouped by universal_gemrate_id + grade (across all sources)."""
    print(
        f"\nGenerating {n_sales_back} lag features (grouped by universal_gemrate_id + grade)..."
    )
    df = df.sort_values(["universal_gemrate_id", "grade", "date"]).reset_index(
        drop=True
    )

    feature_cols = ["price", "half_grade"]
    # Add available grade_co columns
    grade_co_cols = [c for c in df.columns if c.startswith("grade_co_")]
    feature_cols.extend(grade_co_cols)

    new_columns = []

    for n in range(1, n_sales_back + 1):
        suffix = f"prev_{n}"
        for col in feature_cols:
            new_col = f"{suffix}_{col}"
            df[new_col] = df.groupby(["universal_gemrate_id", "grade"])[col].shift(n)
            new_columns.append(new_col)

        days_col = f"{suffix}_days_ago"
        prev_date = df.groupby(["universal_gemrate_id", "grade"])["date"].shift(n)
        df[days_col] = (df["date"] - prev_date).dt.days
        new_columns.append(days_col)

        # Also track source of previous sale
        source_col = f"{suffix}_source_ebay"
        df[source_col] = df.groupby(["universal_gemrate_id", "grade"])[
            "source_ebay"
        ].shift(n)
        new_columns.append(source_col)

    print(f"  → Created {len(new_columns)} lag features")
    return df, new_columns


# ==============================================================================
# MAIN EXPLORATION
# ==============================================================================


def main():
    if not MONGO_URL:
        raise ValueError("MONGO_URL environment variable is not set")

    print("Connecting to MongoDB...")
    client = MongoClient(MONGO_URL)
    db = client[DB_NAME]

    # Step 1: Compare schemas
    print("\n" + "=" * 60)
    print("STEP 1: Schema Comparison")
    print("=" * 60)
    compare_schemas(db)

    # Step 2: Load data (limited for exploration)
    print("\n" + "=" * 60)
    print("STEP 2: Load Sample Data")
    print("=" * 60)

    ebay_raw = load_ebay_data(db[EBAY_COLLECTION], limit=50000)
    pwcc_raw = load_pwcc_data(db[PWCC_COLLECTION], limit=50000)

    # Step 3: Clean data
    print("\n" + "=" * 60)
    print("STEP 3: Clean Data")
    print("=" * 60)

    ebay_clean = clean_ebay_dataframe(ebay_raw)
    pwcc_clean = clean_pwcc_dataframe(pwcc_raw)

    # Step 4: Merge sources
    print("\n" + "=" * 60)
    print("STEP 4: Merge Sources")
    print("=" * 60)

    merged = merge_sources(ebay_clean, pwcc_clean)

    # Step 5: Feature engineering
    print("\n" + "=" * 60)
    print("STEP 5: Feature Engineering")
    print("=" * 60)

    merged = calculate_seller_popularity(merged)
    merged, lag_cols = create_previous_sale_features(merged, n_sales_back=3)

    # Step 6: Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print(f"\nTotal rows: {len(merged)}")
    print(f"\nSource breakdown:")
    print(merged["source"].value_counts())

    print(f"\nUnique universal_gemrate_ids: {merged['universal_gemrate_id'].nunique()}")

    print(f"\nDate range: {merged['date'].min()} to {merged['date'].max()}")

    print(f"\nGrade distribution:")
    print(merged["grade"].value_counts().sort_index())

    print(f"\nFeature columns ({len(merged.columns)} total):")
    for col in sorted(merged.columns):
        print(f"  - {col}")

    # Check data quality for lag features
    print(f"\nLag feature fill rates:")
    for col in lag_cols[:5]:  # Show first 5
        filled = merged[col].notna().sum()
        pct = filled / len(merged) * 100
        print(f"  {col}: {pct:.1f}% filled")

    # Save sample for inspection
    output_file = "merged_sample_data.csv"
    merged.head(1000).to_csv(output_file, index=False)
    print(f"\nSaved sample to {output_file}")

    return merged


if __name__ == "__main__":
    merged_df = main()
