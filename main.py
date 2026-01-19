"""
Main Pipeline Orchestrator

Pipeline Order:
1. Download data from MongoDB + API
3. Feature prep (creates features_prepped.csv for embedding training)
2. Train unified embedding model (contrastive learning on metadata + sales patterns)
5. Neighbor search using 768-dim embeddings
6-10. Neighbor features, model training, inference, QA, upload
"""

import constants
from step_1 import run_step_1
from step_2 import run_step_2
from step_3 import run_step_3
from step_5 import run_step_5
from step_6 import run_step_6
from step_7 import run_step_7
from step_8 import run_step_8
from step_9 import run_step_9
from step_10 import run_step_10

if __name__ == "__main__":
    # Step 1: Download sales data from MongoDB and index from API
    if constants.RUN_STEP_1_DOWNLOAD:
        run_step_1()

    # Step 2: Feature prep (must run before step 3 - creates training data)
    if constants.RUN_STEP_2_FEATURE_PREP:
        run_step_2()

    # Step 3: Train unified embedding model
    if constants.RUN_STEP_3_EMBEDDING:
        run_step_3()

    # Step 5: Neighbor search using unified embeddings
    if constants.RUN_STEP_5_NEIGHBOR_SEARCH:
        run_step_5()

    # Step 6: Add neighbor price features
    if constants.RUN_STEP_6_NEIGHBOR_PRICES:
        run_step_6()

    # Step 7: Train XGBoost model
    if constants.RUN_STEP_7_TRAIN_MODEL:
        run_step_7()

    # Step 8: Run inference
    if constants.RUN_STEP_8_INFERENCE:
        run_step_8()

    # Step 9: QA and sorting
    if constants.RUN_STEP_9_QA:
        run_step_9()

    # Step 10: Upload predictions
    if constants.RUN_STEP_10_UPLOAD:
        run_step_10()
