import os
import torch
from dotenv import load_dotenv

load_dotenv()

# Pipeline Step Flags
RUN_STEP_1_DOWNLOAD = False
RUN_STEP_2_FEATURE_PREP = False
RUN_STEP_3_EMBEDDING = False
RUN_STEP_5_NEIGHBOR_SEARCH = False
RUN_STEP_6_NEIGHBOR_PRICES = False
RUN_STEP_7_TRAIN_MODEL = False
RUN_STEP_8_INFERENCE = True
RUN_STEP_9_QA = True
RUN_STEP_10_UPLOAD = False

# Step 1: Data Download
S1_MONGO_URL = os.getenv("MONGO_URL") or os.getenv("MONGO_URI")
S1_DB_NAME = "gemrate"
S1_EBAY_COLLECTION = "ebay_graded_items"
S1_PWCC_COLLECTION = "pwcc_graded_items"
S1_INDEX_API_URL = "https://price.collectorcrypt.com/api/indexes/modern"
S1_EBAY_MARKET_FILE = "market_ebay.csv"
S1_PWCC_MARKET_FILE = "market_pwcc.csv"
S1_INDEX_FILE = "index.csv"
S1_MONGO_MAX_TIME_MS = 6000000

# Step 2: Feature Prep
S2_INPUT_CATALOG_FILE = "gemrate_pokemon_catalog_20260108.csv"
S2_NUMBER_OF_BIDS_FILTER = 5
S2_N_SALES_BACK = 5
S2_WEEKS_BACK_LIST = [1, 2, 3, 4]
S2_HISTORICAL_DATA_FILE = "historical_data.parquet"
S2_TODAY_DATA_FILE = "today_data.parquet"
S2_FEATURES_PREPPED_FILE = "features_prepped.csv"
S2_INDEX_EMA_SHORT_SPAN = 12
S2_INDEX_EMA_LONG_SPAN = 26
S2_BATCH_SIZE = 1000
S2_LOWEST_GRADE = 7
S2_HIGHEST_GRADE = 11

# Step 3: Unified Embedding (Contrastive Learning)
S3_OUTPUT_EMBEDDINGS_FILE = "card_embeddings.parquet"
S3_MODEL_CHECKPOINT_FILE = "final_encoder.pt"
S3_MODEL_NAME = "BAAI/bge-m3"
S3_EMBEDDING_DIM = 768
S3_ENCODER_EPOCHS = 20
S3_ENCODER_BATCH_SIZE = 512
S3_ENCODER_LR = 5e-4
S3_ENCODER_TAU = 0.07
S3_ENCODER_VAL_SPLIT = 0.1
S3_ENCODER_HIDDEN_1 = 2048
S3_ENCODER_HIDDEN_2 = 1024
S3_DROPOUT_P = 0.1
S3_HASH_N_FEATURES = 2**12
S3_SVD_N_COMPONENTS = 256
S3_OHE_MIN_FREQUENCY = 10
S3_VOLUME_CAP_PER_WEEK = 200.0
S3_EMBED_BATCH_SIZE = 4096

# Step 5: Neighbor Search
S5_N_NEIGHBORS_PREPARE = 50
S5_OUTPUT_NEIGHBORS_FILE = "neighbors.parquet"

# Step 6: Neighbor Features
S6_N_NEIGHBORS = 3
S6_N_NEIGHBOR_SALES = 3
S6_BATCH_SIZE = 100000

# Step 7: Model Training
S7_OUTPUT_MODEL_FILE = "xgb_model.json"

# Step 8: Inference
S8_PREDICTIONS_FILE = "predictions.parquet"

# Step 9: QA
S9_OUTPUT_FILE = "predictions_sorted.parquet"

# Step 10: Upload
S10_PREDICTIONS_COLLECTION = "predictions"
S10_INSERTION_BATCH_SIZE = 100000

# Random Seed
RANDOM_SEED = 42

# Device Selection
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
