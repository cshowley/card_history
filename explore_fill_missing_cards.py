"""
Exploratory script to fill missing cards from sales data.

Problem: gemrate_pokemon_cards is incomplete - some cards with sales history aren't in it.
Solution: Extract card metadata from sales data for any missing universal_gemrate_ids.

Steps:
1. Download all cards from gemrate_pokemon_cards
2. Download sales from ebay_graded_items and pwcc_graded_items (with parsed_description)
3. Find universal_gemrate_ids in sales but NOT in cards
4. Fill missing cards using parsed_description from sales
5. Output combined dataframe for embedding training
"""

import os
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

# ==============================================================================
# CONFIGURATION
# ==============================================================================

MONGO_URL = os.getenv("MONGO_URL") or os.getenv("MONGO_URI")
DB_NAME = "gemrate"
CARDS_COLLECTION = "gemrate_pokemon_cards"
EBAY_COLLECTION = "ebay_graded_items"
PWCC_COLLECTION = "pwcc_graded_items"

OUTPUT_FILE = "all_cards_for_embedding.csv"

# ==============================================================================
# DATA LOADING
# ==============================================================================


def load_cards_from_mongo(db):
    """Load all cards from gemrate_pokemon_cards."""
    print("\n" + "=" * 60)
    print("Loading cards from gemrate_pokemon_cards...")
    print("=" * 60)

    collection = db[CARDS_COLLECTION]
    count = collection.count_documents({})
    print(f"Total documents in collection: {count:,}")

    # Load all cards
    cards = pd.json_normalize(list(collection.find({})))

    if cards.empty:
        print("No cards found!")
        return pd.DataFrame()

    # Keep relevant columns
    columns_to_keep = [
        "GEMRATE_ID",
        "CATEGORY",
        "YEAR",
        "SET_NAME",
        "NAME",
        "PARALLEL",
        "CARD_NUMBER",
    ]

    # Check which columns exist
    available_cols = [c for c in columns_to_keep if c in cards.columns]
    cards = cards[available_cols].copy()

    # Standardize GEMRATE_ID
    cards["GEMRATE_ID"] = cards["GEMRATE_ID"].astype(str)
    cards = cards.drop_duplicates(subset=["GEMRATE_ID"])

    print(f"  → Loaded {len(cards)} unique cards")
    print(f"  → Columns: {list(cards.columns)}")

    return cards


def load_ebay_sales_with_metadata(db, limit=None):
    """Load eBay sales with parsed_description metadata."""
    print("\nLoading eBay sales with metadata...")

    collection = db[EBAY_COLLECTION]

    pipeline = [
        {
            "$match": {
                "gemrate_data.universal_gemrate_id": {"$exists": True, "$ne": ""},
            }
        },
        {
            "$project": {
                "gemrate_data.universal_gemrate_id": 1,
                "gemrate_data.parsed_description.category": 1,
                "gemrate_data.parsed_description.year": 1,
                "gemrate_data.parsed_description.set_name": 1,
                "gemrate_data.parsed_description.name": 1,
                "gemrate_data.parsed_description.parallel": 1,
                "gemrate_data.parsed_description.card_number": 1,
                "_id": 0,
            }
        },
    ]

    if limit:
        pipeline.append({"$limit": limit})

    results = list(collection.aggregate(pipeline, maxTimeMS=600000, allowDiskUse=True))
    df = pd.json_normalize(results)

    print(f"  → Loaded {len(df)} eBay sales")
    if not df.empty:
        print(f"  → Columns: {list(df.columns)}")

    return df


def load_pwcc_sales_with_metadata(db, limit=None):
    """Load PWCC sales with parsed_description metadata."""
    print("\nLoading PWCC sales with metadata...")

    collection = db[PWCC_COLLECTION]

    pipeline = [
        {
            "$match": {
                "gemrate_data.universal_gemrate_id": {"$exists": True, "$ne": ""},
            }
        },
        {
            "$project": {
                "gemrate_data.universal_gemrate_id": 1,
                "gemrate_data.parsed_description.category": 1,
                "gemrate_data.parsed_description.year": 1,
                "gemrate_data.parsed_description.set_name": 1,
                "gemrate_data.parsed_description.name": 1,
                "gemrate_data.parsed_description.parallel": 1,
                "gemrate_data.parsed_description.card_number": 1,
                "_id": 0,
            }
        },
    ]

    if limit:
        pipeline.append({"$limit": limit})

    results = list(collection.aggregate(pipeline, maxTimeMS=600000, allowDiskUse=True))
    df = pd.json_normalize(results)

    print(f"  → Loaded {len(df)} PWCC sales")
    if not df.empty:
        print(f"  → Columns: {list(df.columns)}")

    return df


def extract_unique_cards_from_sales(ebay_df, pwcc_df):
    """Extract unique cards from sales data."""
    print("\n" + "=" * 60)
    print("Extracting unique cards from sales...")
    print("=" * 60)

    # Combine both sources
    combined = pd.concat([ebay_df, pwcc_df], ignore_index=True)
    print(f"Combined sales: {len(combined)} rows")

    # Rename columns to match card schema
    rename_map = {
        "gemrate_data.universal_gemrate_id": "GEMRATE_ID",
        "gemrate_data.parsed_description.category": "CATEGORY",
        "gemrate_data.parsed_description.year": "YEAR",
        "gemrate_data.parsed_description.set_name": "SET_NAME",
        "gemrate_data.parsed_description.name": "NAME",
        "gemrate_data.parsed_description.parallel": "PARALLEL",
        "gemrate_data.parsed_description.card_number": "CARD_NUMBER",
    }

    # Rename available columns
    for old, new in rename_map.items():
        if old in combined.columns:
            combined = combined.rename(columns={old: new})

    # Ensure GEMRATE_ID exists and is string
    if "GEMRATE_ID" not in combined.columns:
        print("ERROR: No GEMRATE_ID found in sales data!")
        return pd.DataFrame()

    combined["GEMRATE_ID"] = combined["GEMRATE_ID"].astype(str)

    # Drop duplicates - keep first occurrence (any metadata is fine)
    columns_for_dedup = ["GEMRATE_ID"]
    unique_cards = combined.drop_duplicates(subset=columns_for_dedup)

    # Keep only card metadata columns
    card_cols = [
        "GEMRATE_ID",
        "CATEGORY",
        "YEAR",
        "SET_NAME",
        "NAME",
        "PARALLEL",
        "CARD_NUMBER",
    ]
    available_cols = [c for c in card_cols if c in unique_cards.columns]
    unique_cards = unique_cards[available_cols].copy()

    print(f"  → Unique cards from sales: {len(unique_cards)}")

    return unique_cards


def fill_missing_cards(existing_cards, sales_cards):
    """Find and fill missing cards from sales data."""
    print("\n" + "=" * 60)
    print("Finding and filling missing cards...")
    print("=" * 60)

    existing_ids = set(existing_cards["GEMRATE_ID"].astype(str))
    sales_ids = set(sales_cards["GEMRATE_ID"].astype(str))

    missing_ids = sales_ids - existing_ids
    overlap_ids = sales_ids & existing_ids

    print(f"Existing cards in collection: {len(existing_ids):,}")
    print(f"Unique cards in sales: {len(sales_ids):,}")
    print(f"Cards in BOTH: {len(overlap_ids):,}")
    print(f"Cards in sales but NOT in collection: {len(missing_ids):,}")

    if len(missing_ids) == 0:
        print("No missing cards to fill!")
        return existing_cards

    # Filter sales_cards to only missing IDs
    missing_cards = sales_cards[sales_cards["GEMRATE_ID"].isin(missing_ids)].copy()
    print(f"\nMissing cards to add: {len(missing_cards)}")

    # Show sample of missing cards
    print("\nSample missing cards:")
    print(missing_cards.head(10).to_string())

    # Check for empty metadata in missing cards
    for col in ["CATEGORY", "YEAR", "SET_NAME", "NAME"]:
        if col in missing_cards.columns:
            empty_count = (
                missing_cards[col].isna().sum() + (missing_cards[col] == "").sum()
            )
            pct = empty_count / len(missing_cards) * 100
            print(f"  {col}: {empty_count} empty ({pct:.1f}%)")

    # Combine existing + missing
    combined = pd.concat([existing_cards, missing_cards], ignore_index=True)
    combined = combined.drop_duplicates(subset=["GEMRATE_ID"])

    print(f"\nFinal combined cards: {len(combined)}")

    return combined


def main():
    if not MONGO_URL:
        raise ValueError("MONGO_URL environment variable is not set")

    print("Connecting to MongoDB...")
    client = MongoClient(MONGO_URL)
    db = client[DB_NAME]

    # Step 1: Load existing cards
    existing_cards = load_cards_from_mongo(db)

    # Step 2: Load sales with metadata (no limit for full run)
    ebay_sales = load_ebay_sales_with_metadata(db, limit=None)
    pwcc_sales = load_pwcc_sales_with_metadata(db, limit=None)

    # Step 3: Extract unique cards from sales
    sales_cards = extract_unique_cards_from_sales(ebay_sales, pwcc_sales)

    # Step 4: Fill missing cards
    all_cards = fill_missing_cards(existing_cards, sales_cards)

    # Step 5: Save output
    print("\n" + "=" * 60)
    print("Saving output...")
    print("=" * 60)

    all_cards.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(all_cards)} cards to {OUTPUT_FILE}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Original cards from collection: {len(existing_cards)}")
    print(f"Cards added from sales: {len(all_cards) - len(existing_cards)}")
    print(f"Total cards for embedding: {len(all_cards)}")

    return all_cards


if __name__ == "__main__":
    all_cards = main()
