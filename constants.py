import os
import torch
from dotenv import load_dotenv

load_dotenv()

RUN_STEP_1_DOWNLOAD = False
RUN_STEP_2_TEXT_EMBEDDING = False
RUN_STEP_3_FEATURE_PREP = False
RUN_STEP_4_PRICE_EMBEDDING = False
RUN_STEP_5_NEIGHBOR_SEARCH = False
RUN_STEP_6_NEIGHBOR_PRICES = False
RUN_STEP_7_TRAIN_MODEL = False
RUN_STEP_8_INFERENCE = False
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
S3_WEEKS_BACK_LIST = [1, 2, 3, 4]
S3_HISTORICAL_DATA_FILE = "historical_data.parquet"
S3_TODAY_DATA_FILE = "today_data.parquet"
S3_INDEX_EMA_SHORT_SPAN = 12
S3_INDEX_EMA_LONG_SPAN = 26
S3_BATCH_SIZE = 1000
S3_LOWEST_GRADE = 7
S3_HIGHEST_GRADE = 11

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

S8_PREDICTIONS_FILE = "predictions.parquet"

S9_OUTPUT_FILE = "predictions_sorted.parquet"

S10_PREDICTIONS_COLLECTION = "predictions"
S10_INSERTION_BATCH_SIZE = 100000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
