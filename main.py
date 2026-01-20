import constants
from step_1 import run_step_1
from step_2 import run_step_2
from step_3 import run_step_3
from step_4 import run_step_4
from step_5 import run_step_5
from step_6 import run_step_6
from step_7 import run_step_7
from step_8 import run_step_8
from step_9 import run_step_9
from step_10 import run_step_10
from data_integrity import save_to_mongo

if __name__ == "__main__":
    if constants.RUN_STEP_1_DOWNLOAD:
        run_step_1()

    if constants.RUN_STEP_2_TEXT_EMBEDDING:
        run_step_2()

    if constants.RUN_STEP_3_FEATURE_PREP:
        run_step_3()

    if constants.RUN_STEP_4_PRICE_EMBEDDING:
        run_step_4()

    if constants.RUN_STEP_5_NEIGHBOR_SEARCH:
        run_step_5()

    if constants.RUN_STEP_6_NEIGHBOR_PRICES:
        run_step_6()

    if constants.RUN_STEP_7_TRAIN_MODEL:
        run_step_7()

    if constants.RUN_STEP_8_INFERENCE:
        run_step_8()

    if constants.RUN_STEP_9_QA:
        run_step_9()

    if constants.RUN_STEP_10_UPLOAD:
        run_step_10()

    # Save data integrity metrics to MongoDB
    print("\nSaving data integrity metrics...")
    save_to_mongo()
    print("Pipeline complete.")
