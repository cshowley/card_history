import os

import torch
from dotenv import load_dotenv

load_dotenv()

RUN_STEP_1_DOWNLOAD = True
RUN_STEP_2_TEXT_EMBEDDING = False
RUN_STEP_3_FEATURE_PREP = True
RUN_STEP_4_PRICE_EMBEDDING = True
RUN_STEP_5_NEIGHBOR_SEARCH = True
RUN_STEP_6_NEIGHBOR_PRICES = True
RUN_STEP_7_TRAIN_MODEL = True
RUN_STEP_8_INFERENCE = True
RUN_STEP_9_QA = True
RUN_STEP_10_UPLOAD = True

S1_MONGO_URL = os.getenv("MONGO_URL") or os.getenv("MONGO_URI")
S1_DB_NAME = "gemrate"
S1_EBAY_COLLECTION = "ebay_graded_items"
S1_PWCC_COLLECTION = "pwcc_graded_items"
S1_INDEX_API_URL = "https://price.collectorcrypt.com/api/indexes/modern"
S1_EBAY_MARKET_FILE = "market_ebay.csv"
S1_PWCC_MARKET_FILE = "market_pwcc.csv"
S1_INDEX_FILE = "index.csv"
S1_MONGO_MAX_TIME_MS = 6000000

S2_INPUT_CATALOG_FILE = "gemrate_pokemon_catalog_20260108.csv"
S2_OUTPUT_EMBEDDINGS_FILE = "text_embeddings.parquet"
S2_MODEL_NAME = "BAAI/bge-m3"

S3_NUMBER_OF_BIDS_FILTER = 5
S3_N_SALES_BACK = 5
S3_WEEKS_BACK_LIST = [0, 1, 2, 3, 4]
S3_HISTORICAL_DATA_FILE = "historical_data.parquet"
S3_TODAY_DATA_FILE = "today_data.parquet"
S3_HISTORICAL_DATA_LOG_FILE = "historical_data_log.parquet"
S3_TODAY_DATA_LOG_FILE = "today_data_log.parquet"
S3_INDEX_EMA_SHORT_SPAN = 12
S3_INDEX_EMA_LONG_SPAN = 26
S3_BATCH_SIZE = 1000
S3_LOWEST_GRADE = 7
S3_HIGHEST_GRADE = 10
S3_START_DATE = "2025-09-01"
S3_END_DATE = None  # "2026-01-22"

S4_WINDOW_SIZE = 4
S4_BATCH_SIZE = 1024
S4_EPOCHS = 100
S4_PRICE_EMBEDDING_DIM = 32
S4_LEARNING_RATE = 0.0001
S4_OUTPUT_PRICE_VECS_FILE = "price_embeddings.parquet"


S5_N_NEIGHBORS_PREPARE = 50
S5_OUTPUT_NEIGHBORS_FILE = "neighbors.parquet"


S6_N_NEIGHBORS = 3
S6_N_NEIGHBOR_SALES = 3
S6_BATCH_SIZE = 100000

S7_OUTPUT_MODEL_FILE = "xgb_model.json"

S7_CUSTOM_BIN_EDGES = [0.01, 32, 64, 128, 1024, float("inf")]
S7_CUSTOM_BIN_LABELS = ["$0.01-$32", "$32-$64", "$64-$128", "$128-$1024", "$1024+"]

S7_PRIMARY_MODEL_OBJECTIVE = "reg:gamma"
S7_PRIMARY_MODEL_DATASET = "normal"
S7_PRIMARY_MODEL_OUTPUT_DIR = "model/output/primary"
S7_PRIMARY_MODEL_RESULTS_DIR = "model/results/gamma"

S7_FALLBACK_MODEL_OBJECTIVE = "reg:gamma"
S7_FALLBACK_MODEL_DATASET = "normal"
S7_FALLBACK_MODEL_OUTPUT_DIR = "model/output/fallback"
S7_FALLBACK_MODEL_RESULTS_DIR = "model/results/gamma"

S7_EXCLUDE_COLS = ["gemrate_id", "date", "price", "_row_id"]

S8_PREDICTIONS_FILE = "predictions.parquet"

S9_OUTPUT_FILE = "predictions_sorted.parquet"

S10_PREDICTIONS_COLLECTION = "predictions"
S10_INSERTION_BATCH_SIZE = 100000
S10_UPSERT = True


def get_dataset_files(dataset_type: str) -> dict:
    """
    Return dataset file paths based on dataset type.

    Args:
        dataset_type: Either "normal" or "log"

    Returns:
        Dictionary with keys:
        - historical_file: Path to the historical data parquet file
        - today_file: Path to the today data parquet file

    Raises:
        ValueError: If the dataset type is not recognized.
    """
    if dataset_type == "normal":
        return {
            "historical_file": S3_HISTORICAL_DATA_FILE,
            "today_file": S3_TODAY_DATA_FILE,
        }
    elif dataset_type == "log":
        return {
            "historical_file": S3_HISTORICAL_DATA_LOG_FILE,
            "today_file": S3_TODAY_DATA_LOG_FILE,
        }
    else:
        raise ValueError(
            f"Unknown dataset type: {dataset_type}. Must be 'normal' or 'log'."
        )


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
