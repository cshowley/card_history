import pandas as pd
from pymongo import MongoClient
import constants
from tqdm import tqdm


def run_step_10():
    print("Starting Step 10: Upload Predictions to MongoDB...")

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

    print("Creating index on 'gemrate_id'...")
    collection.create_index("gemrate_id")

    batch_size = constants.S10_INSERTION_BATCH_SIZE
    print(f"Inserting into MongoDB '{collection_name}' in batches of {batch_size}...")

    inserted_count = 0

    for i in tqdm(range(0, len(df), batch_size), desc="Uploading Batches"):
        batch = df.iloc[i : i + batch_size].copy()
        records = batch.to_dict("records")
        if records:
            collection.insert_many(records)
            inserted_count += len(records)

    print(
        f"Step 10 Complete. Uploaded {inserted_count} documents to collection '{collection_name}'."
    )


if __name__ == "__main__":
    run_step_10()
