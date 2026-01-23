import time

import pandas as pd
from pymongo import MongoClient, ReplaceOne
from tqdm import tqdm

import constants
from data_integrity import get_tracker


def run_step_10():
    print("Starting Step 10: Upload Predictions to MongoDB...")
    start_time = time.time()
    tracker = get_tracker()

    if not constants.S1_MONGO_URL:
        raise ValueError("MONGO_URL environment variable is not set")

    input_file = constants.S9_OUTPUT_FILE
    collection_name = constants.S10_PREDICTIONS_COLLECTION

    print(f"Reading predictions from {input_file}...")
    try:
        df = pd.read_parquet(input_file)
    except FileNotFoundError:
        print(f"File {input_file} not found. Please run Step 8 first.")
        return

    print(f"Loaded {len(df)} predictions.")

    client = MongoClient(constants.S1_MONGO_URL)
    db = client[constants.S1_DB_NAME]

    collection = db[collection_name]

    print(f"Deleting all documents in '{collection_name}'...")
    collection.delete_many({})

    batch_size = constants.S10_INSERTION_BATCH_SIZE
    print(f"Upserting into MongoDB '{collection_name}' in batches of {batch_size}...")

    inserted_count = 0

    for i in tqdm(range(0, len(df), batch_size), desc="Uploading Batches"):
        batch = df.iloc[i : i + batch_size].copy()
        records = batch.to_dict("records")

        operations = []
        for record in records:
            filter_query = {
                "gemrate_id": record["gemrate_id"],
                "grading_company": record["grading_company"],
                "grade": record["grade"],
                "half_grade": record["half_grade"],
            }
            operations.append(ReplaceOne(filter_query, record, upsert=True))

        if operations:
            collection.bulk_write(operations)
            inserted_count += len(operations)

    print("Creating index on 'gemrate_id'...")
    collection.create_index("gemrate_id")

    # Data Integrity Tracking
    duration = time.time() - start_time
    tracker.add_metric(
        id="s10_documents_uploaded",
        title="Documents Uploaded",
        value=inserted_count,
    )
    tracker.add_metric(
        id="s10_duration",
        title="Step 10 Duration",
        value=round(duration, 1),
    )

    print(
        f"Step 10 Complete. Uploaded {inserted_count} documents to collection '{collection_name}'."
    )


if __name__ == "__main__":
    run_step_10()
