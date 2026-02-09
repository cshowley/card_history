# AGENTS.md - Collector Crypt Data Pipeline

## Project Overview

Python 3.13 ML pipeline for predicting graded Pokemon trading card prices.
10 sequential steps: data download (MongoDB) -> embeddings (text + price) ->
nearest-neighbor search (FAISS) -> feature engineering -> XGBoost training ->
inference -> monotonicity enforcement -> upload predictions to MongoDB.

## Tech Stack

- **Language**: Python 3.13
- **Package manager**: `uv` (with `uv.lock`)
- **ML**: XGBoost (regression), PyTorch (LSTM autoencoder), SentenceTransformers (BAAI/bge-m3)
- **Similarity search**: FAISS (CPU on macOS, GPU on Linux)
- **Data**: pandas, NumPy, PyArrow (Parquet files for inter-step data)
- **Database**: MongoDB via pymongo
- **GPU**: CUDA/CuPy on Linux, MPS on macOS (auto-detected in `constants.py`)
- **Visualization**: matplotlib, seaborn

## Build and Run Commands

```bash
# Install dependencies
uv sync

# Run the full pipeline
uv run python main.py

# Run a single step (set flags in constants.py)
# Toggle RUN_STEP_N_* booleans to True/False, then:
uv run python main.py

# Run individual step files directly (some support __main__)
uv run python step_7.py
uv run python step_8.py
uv run python step_9.py
uv run python step_10.py
uv run python prediction_query.py

# Hyperparameter search (separate workflow)
uv run python model/run_pipeline.py

# Model evaluation
uv run python model/model_test.py
```

## Testing

Tests use **pytest** (dev dependency) and verify data integrity metrics accuracy.
`model/model_test.py` is a model evaluation script, not a unit test.

```bash
# Install dev dependencies (includes pytest)
uv sync --group dev

# Run the full test suite (48 tests, ~2 seconds)
uv run pytest tests/ -v

# Run a single test file
uv run pytest tests/test_step_9_metrics.py -v

# Run a single test
uv run pytest tests/test_step_9_metrics.py::TestStep9Monotonicity::test_violation_count_with_violations -v
```

Test files cover steps 1, 2, 3, 4, 5, 8, and 9. Tests are non-invasive: all
mocking is temporary (restored automatically via `unittest.mock.patch`), the
global tracker is reset between tests, and temp files are auto-cleaned.

### Adding Tests for New Metrics

Every `tracker.add_*` call in the codebase is cataloged in
`tests/METRIC_COVERAGE.md` with its widget ID, source location, test file,
test method name, and coverage status. **When adding a new metric, consult that
file first** to find the right test file and pattern to follow, then add a row.

Two test strategies are used:

| Strategy | Test files | How it works |
|----------|-----------|--------------|
| **Mock-and-intercept** | steps 1, 2, 5, 9 | Mocks I/O (MongoDB, HTTP, ML models), calls real `run_step_N()`, inspects tracker |
| **Replicate-and-verify** | steps 3, 4, 8 | Builds synthetic DataFrame, runs same pandas/numpy formula, pushes to tracker, asserts |

**Mock-and-intercept** (e.g., adding `s1_new_metric` to step 1):
```python
# tests/test_step_1_metrics.py
def test_new_metric(self, fresh_tracker, tmp_path):
    ebay_df, _ = make_ebay_df(n_rows=10, n_before_cutoff=0)
    pwcc_df, _ = make_pwcc_df(n_rows=5, n_before_cutoff=0)
    _run_step_1_with_mocks(ebay_df, pwcc_df, tmp_path)
    w = find_widget(get_tracker(), "s1_new_metric")
    assert w is not None
    assert w["value"] == expected_value
```

**Replicate-and-verify** (e.g., adding `s8_prediction_p90` to step 8):
```python
# tests/test_step_8_metrics.py -- add to _simulate_step8_metrics():
pred_p90 = float(np.percentile(preds_arr, 90))
tracker.add_metric(id="s8_prediction_p90", title="P90", value=round(pred_p90, 4))

# Then add a test:
def test_p90(self, fresh_tracker):
    predictions = [10.0, 20.0, 30.0, 40.0, 50.0]
    _simulate_step8_metrics(fresh_tracker, predictions)
    assert find_widget(fresh_tracker, "s8_prediction_p90")["value"] == 46.0
```

If the new metric needs synthetic data with specific properties, add a parameter
to the appropriate factory in `tests/helpers/synthetic_data.py`.

## Linting and Formatting

**No linting or formatting tools are configured.** There is no ruff, black, flake8,
pylint, mypy, isort, or pre-commit setup. Follow the existing code style described
below when making changes.

## Environment Variables

Required in `.env` (loaded via python-dotenv):
- `MONGO_URL` or `MONGO_URI` - MongoDB connection string (read-only, for data download)
- `MONGO_URI_RW` - MongoDB connection string (read-write, for uploading predictions and metrics)

## Project Structure

```
main.py              # Pipeline orchestrator - runs steps 1-10 sequentially
constants.py         # All configuration: file paths, DB settings, step toggles, hyperparams
data_integrity.py    # Global DataIntegrityTracker singleton, saved to MongoDB
step_1.py            # Download sales data from eBay/PWCC via MongoDB + market index
step_2.py            # Generate text embeddings (SentenceTransformer)
step_3.py            # Clean/merge sales data, generate features
step_4.py            # Train LSTM autoencoder for price embeddings
step_5.py            # Nearest-neighbor search (FAISS) using text + price embeddings
step_6.py            # Enhance dataset with neighbor sales history features
step_7.py            # Train XGBoost models (base/lower/upper) in parallel
step_8.py            # Run inference on today's data
step_9.py            # Sort predictions, enforce monotonicity constraints
step_10.py           # Upload predictions to MongoDB
prediction_query.py  # CLI utility to query predictions
model/               # Hyperparameter search and model evaluation
  model_setup.py     # Data split, hyperparameter search launcher
  hyperparameter_search_worker.py  # GPU worker for XGBoost search
  model_test.py      # Evaluate best model on test set
  run_pipeline.py    # Entry point for model search workflow
tests/               # Data integrity metric sanity-check tests (pytest)
  conftest.py        # Shared fixtures: fresh_tracker, find_widget helper
  helpers/
    synthetic_data.py  # Factory functions for controlled test DataFrames
  test_step_1_metrics.py  # Step 1: record counts, anomalies, missing data, prices
  test_step_2_metrics.py  # Step 2: text quality, embedding output, failure tracking
  test_step_3_metrics.py  # Step 3: sales charts, histograms, concentration tables
  test_step_4_metrics.py  # Step 4: input validation, ID counts, error tracking
  test_step_5_metrics.py  # Step 5: neighbor coverage, DB size, pair counts
  test_step_8_metrics.py  # Step 8: prediction stats, model collapse detection
  test_step_9_metrics.py  # Step 9: monotonicity violations, QA outliers
```

## Code Style Guidelines

### Formatting
- **Indentation**: 4 spaces, no tabs
- **Line length**: ~100 characters soft limit
- **Quotes**: Double quotes (`"`) exclusively
- **Trailing commas**: Yes, in all multiline constructs
- **Semicolons**: Never
- **String formatting**: f-strings everywhere
- **Blank lines**: 2 between top-level definitions (PEP 8), 1 within functions

### Naming Conventions
- **Functions**: `snake_case` - e.g., `run_step_1()`, `extract_numbers()`
- **Variables**: `snake_case` - e.g., `ebay_df`, `best_val_loss`
- **Classes**: `PascalCase` - e.g., `LSTMAutoencoder`, `DataIntegrityTracker`
- **Constants**: `UPPER_SNAKE_CASE` - e.g., `RUN_STEP_1_DOWNLOAD`, `DEVICE`
- **Files**: `snake_case` - e.g., `step_1.py`, `data_integrity.py`
- **Step-scoped helpers**: Prefix with `sN_` - e.g., `s3_clean_grade()`, `s5_build_search_matrices()`
- **Step-scoped constants**: Prefix with `SN_` - e.g., `S3_BATCH_SIZE`, `S7_OUTPUT_MODEL_FILE`
- **Private variables**: Leading underscore - e.g., `_tracker`, `_pipeline_start`

### Imports
- **Absolute imports only** - never use relative imports (`from . import ...`)
- **Group order**: standard library, third-party, local project (separated by blank lines)
- Prefer `import X` over `from X import Y` for stdlib/third-party (exceptions: specific classes)
- Common aliases: `import numpy as np`, `import pandas as pd`

### Type Annotations
- Type annotations are **minimal** - most functions have none
- `data_integrity.py` is the exception with annotations on params and returns
- No type checker (mypy/pyright) is configured
- When adding new code to `data_integrity.py`, include type annotations for consistency
- For other files, type annotations are optional but not required

### Error Handling
- Raise built-in exceptions with descriptive messages:
  `raise ValueError("MONGO_URL environment variable is not set")`
- Use `try/except` with tracker logging and re-raise for recoverable operations:
  ```python
  try:
      result = risky_operation()
  except Exception as e:
      tracker.add_error(f"Operation failed: {e}", step="step_N")
      print(f"ERROR: Operation failed: {e}")
      raise
  ```
- Use `try/finally` for resource cleanup (file handles, DB connections)
- Guard with early returns for missing files:
  ```python
  if not os.path.exists(path):
      print(f"File not found: {path}")
      return
  ```
- No custom exception classes - use built-in `ValueError`, `FileNotFoundError`, `RuntimeError`

### Docstrings and Comments
- **Google-style docstrings** with `Args:` sections when docstrings are present
- Docstrings are sparse - mainly in `data_integrity.py`; most step functions lack them
- Use inline comments as section headers within long functions:
  ```python
  # Data Integrity Tracking
  # Price outlier detection
  ```
- Module-level docstrings for utility modules (see `data_integrity.py`)

### File Organization Pattern
Each `step_N.py` follows this structure:
1. Imports (grouped: stdlib, third-party, local)
2. Module-level constants (if any)
3. Helper functions (prefixed with `sN_`)
4. Main `run_step_N()` function
5. `if __name__ == "__main__":` block (optional)

### Configuration
- All configuration lives in `constants.py` as module-level variables
- Step toggles are booleans: `RUN_STEP_N_*`
- File paths, batch sizes, hyperparameters are constants prefixed by step number
- Environment variables loaded via `python-dotenv` in `constants.py`
- Do NOT scatter configuration across step files

### Data Integrity Tracking
- Use `get_tracker()` to get the global `DataIntegrityTracker` singleton
- Track metrics via `tracker.add_metric(id=..., title=..., value=...)`
- Track errors via `tracker.add_error(message, step="step_N")`
- Track tables via `tracker.add_table(id=..., title=..., columns=..., data=...)`
- Track charts via `tracker.add_chart(id=..., title=..., chart_type=..., columns=..., data=...)`
- All widget IDs should be descriptive and prefixed with step: `s1_total_records`

### Parallelism
- No async/await - all I/O is synchronous
- `multiprocessing.Process` for XGBoost training (step_7)
- `subprocess.Popen` for hyperparameter search workers
- `concurrent.futures.ThreadPoolExecutor` for inference (step_8)

### Data Flow
- Steps communicate via Parquet files on disk (defined in `constants.py`)
- MongoDB for initial data download and final prediction upload
- No in-memory state shared between steps
