from datetime import datetime, timedelta

import pandas as pd
import requests
from pymongo import MongoClient

import constants


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
    if not constants.S1_MONGO_URL:
        raise ValueError("MONGO_URL environment variable is not set")

    client = MongoClient(constants.S1_MONGO_URL)
    db = client[constants.S1_DB_NAME]

    # Calculate cutoff date for 7-day filter
    cutoff_date = datetime.now() - timedelta(days=7)
    cutoff_date_str = cutoff_date.strftime("%Y-%m-%d")

    print("Downloading eBay sales...")
    ebay_collection = db[constants.S1_EBAY_COLLECTION]
    ebay_pipeline = [
        {
            "$match": {
                "$or": [
                    {"gemrate_data.universal_gemrate_id": {"$exists": True, "$ne": ""}},
                    {"gemrate_data.gemrate_id": {"$exists": True, "$ne": ""}},
                ],
                "item_data.date": {"$exists": True, "$gte": cutoff_date_str},
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
    print(f"  → eBay rows loaded: {len(ebay_df)}")
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
                "api_response.soldDate": {"$exists": True, "$gte": cutoff_date_str},
                "api_response.purchasePrice": {"$exists": True},
                "api_response.auctionType": {"$in": ["WEEKLY", "PREMIER"]},
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
    print(f"  → PWCC rows loaded: {len(pwcc_df)}")
    pwcc_df.to_csv(constants.S1_PWCC_MARKET_FILE, index=False)

    index_df = s1_fetch_index_data(constants.S1_INDEX_API_URL)
    index_df.to_csv(constants.S1_INDEX_FILE, index=False)
    print("Step 1 Complete.")
