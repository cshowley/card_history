# Metric Coverage Registry

Every `tracker.add_metric/table/chart/error` call in the pipeline is listed below
with its test coverage status. When adding a new metric, add a row here and write
a corresponding test. See [How to Add a Test](#how-to-add-a-test-for-a-new-metric)
at the bottom.

**Coverage: 48 tests covering 45 of 81 tracker calls (56%). All high-value
computation metrics are covered; gaps are duration timers, pipeline orchestration,
and steps requiring GPU/MongoDB-RW.**

---

## main.py (0 / 4 covered)

| Widget ID | Type | Source | Test | Status |
|-----------|------|--------|------|--------|
| `pipeline_start_time` | metric | `main.py:21` | -- | not covered (pipeline orchestration) |
| `step_summary` | table | `main.py:88` | -- | not covered (pipeline orchestration) |
| `pipeline_end_time` | metric | `main.py:96` | -- | not covered (pipeline orchestration) |
| `pipeline_total_duration` | metric | `main.py:102` | -- | not covered (timing only) |

---

## Step 1 (11 / 17 covered)

Strategy: **mock-and-intercept** -- mocks MongoClient and requests.get, calls real `run_step_1()`.

| Widget ID | Type | Source | Test File | Test(s) | Status |
|-----------|------|--------|-----------|---------|--------|
| `s1_sales_before_sep_2025` | metric | `step_1.py:150` | `test_step_1_metrics.py` | `test_sales_before_cutoff`, `test_zero_before_cutoff` | covered |
| `s1_dropped_sales_before_sep_2025` | metric | `step_1.py:173` | `test_step_1_metrics.py` | `test_dropped_sales_count`, `test_zero_before_cutoff` | covered |
| `s1_total_records` | metric | `step_1.py:187` | `test_step_1_metrics.py` | `test_total_and_source_counts` | covered |
| `s1_ebay_records` | metric | `step_1.py:193` | `test_step_1_metrics.py` | `test_total_and_source_counts` | covered |
| `s1_pwcc_records` | metric | `step_1.py:199` | `test_step_1_metrics.py` | `test_total_and_source_counts` | covered |
| `s1_anomalies` | table | `step_1.py:214` | `test_step_1_metrics.py` | `test_bid_anomalies` | covered |
| `s1_ebay_missing` | table | `step_1.py:258` | `test_step_1_metrics.py` | `test_ebay_missing_gemrate_id` | covered |
| `s1_pwcc_missing` | table | `step_1.py:302` | `test_step_1_metrics.py` | `test_pwcc_missing_data` | covered |
| `s1_ebay_grades` | chart | `step_1.py:314` | `test_step_1_metrics.py` | `test_ebay_grade_chart_emitted` | covered |
| `s1_extreme_prices_count` | metric | `step_1.py:389` | `test_step_1_metrics.py` | `test_extreme_prices` | covered |
| `s1_marketplace_breakdown` | table | `step_1.py:400` | `test_step_1_metrics.py` | `test_share_percentages` | covered |
| `s1_pwcc_grades` | chart | `step_1.py:327` | -- | not covered (same logic as `s1_ebay_grades`) |
| `s1_most_recent_ebay_date` | metric | `step_1.py:344` | -- | not covered (data freshness, wall-clock dependent) |
| `s1_most_recent_pwcc_date` | metric | `step_1.py:357` | -- | not covered (data freshness, wall-clock dependent) |
| `s1_days_since_last_sale` | metric | `step_1.py:372` | -- | not covered (wall-clock dependent) |
| `s1_duration` | metric | `step_1.py:395` | -- | not covered (timing only) |
| `step_1` error | error | `step_1.py:28` | -- | not covered (requires MONGO_URL to be unset) |

---

## Step 2 (7 / 9 covered)

Strategy: **mock-and-intercept** -- mocks SentenceTransformer, calls real `run_step_2()`.

| Widget ID | Type | Source | Test File | Test(s) | Status |
|-----------|------|--------|-----------|---------|--------|
| `s2_input_rows` | metric | `step_2.py:98` | `test_step_2_metrics.py` | `test_input_rows_after_dedup` | covered |
| `s2_empty_text_count` | metric | `step_2.py:107` | `test_step_2_metrics.py` | `test_empty_text_count` | covered |
| `s2_avg_text_length` | metric | `step_2.py:112` | `test_step_2_metrics.py` | `test_avg_text_length_positive` | covered |
| `s2_embedding_failed` | metric | `step_2.py:130` | `test_step_2_metrics.py` | `test_embedding_failure_tracked` | covered |
| `s2_cards_embedded` | metric | `step_2.py:149` | `test_step_2_metrics.py` | `test_cards_embedded` | covered |
| `s2_embedding_dim` | metric | `step_2.py:154` | `test_step_2_metrics.py` | `test_embedding_dimension` | covered |
| `step_2` error | error | `step_2.py:126` | `test_step_2_metrics.py` | `test_embedding_failure_tracked` | covered |
| `s2_duration` | metric | `step_2.py:159` | -- | not covered (timing only) |
| `s2` download fallback | -- | `step_2.py:43` | -- | not covered (MongoDB download path) |

---

## Step 3 (7 / 13 covered)

Strategy: **replicate-and-verify** -- builds synthetic DataFrames matching the step 3
schema, replicates the pandas groupby/agg computation, pushes to tracker, verifies.

| Widget ID | Type | Source | Test File | Test(s) | Status |
|-----------|------|--------|-----------|---------|--------|
| `sales_per_day` | chart | `step_3.py:434` | `test_step_3_metrics.py` | `test_sales_per_day_chart` | covered |
| `dollar_volume_per_day` | chart | `step_3.py:454` | `test_step_3_metrics.py` | `test_dollar_volume_chart` | covered |
| `median_price_per_day` | chart | `step_3.py:472` | `test_step_3_metrics.py` | `test_median_price_chart` | covered |
| `sales_price_histogram` | chart | `step_3.py:493` | `test_step_3_metrics.py` | `test_histogram_bins` | covered |
| `s3_median_days_between_top_100_sales` | metric | `step_3.py:533` | `test_step_3_metrics.py` | `test_median_days_computation` | covered |
| `sales_concentration_per_day` | table | `step_3.py:549` | `test_step_3_metrics.py` | `test_concentration_data` | covered |
| `sales_grade_histogram` | chart | `step_3.py:510` | -- | not covered (same pattern as `sales_price_histogram`) |
| `sales_comparison` | table | `step_3.py:571` | -- | not covered (date comparison table) |
| `s3_total_cleaned` | metric | `step_3.py:612` | -- | not covered (requires full batch loop with catalog filter) |
| `s3_batch_loop_completed` | metric | `step_3.py:713` | -- | not covered (requires full batch loop) |
| `first_time_sales_per_day` | chart | `step_3.py:725` | -- | not covered (requires full batch loop) |
| `s3_duration` | metric | `step_3.py:738` | -- | not covered (timing only) |

**Note**: The uncovered step 3 metrics (`s3_total_cleaned`, `s3_batch_loop_completed`,
`first_time_sales_per_day`) live inside the batch processing loop (lines 636-734)
which reads 6+ files and writes Parquet incrementally. Mocking the full loop is
possible but fragile. If these metrics become important to test, consider extracting
the metric logic into a helper function.

---

## Step 4 (5 / 11 covered)

Strategy: **replicate-and-verify** -- writes synthetic parquet, replicates input
validation logic. LSTM training metrics are not tested (require real training).

| Widget ID | Type | Source | Test File | Test(s) | Status |
|-----------|------|--------|-----------|---------|--------|
| `s4_input_file_size_mb` | metric | `step_4.py:264` | `test_step_4_metrics.py` | `test_input_file_size_metric` | covered |
| `s4_input_rows` | metric | `step_4.py:277` | `test_step_4_metrics.py` | `test_input_rows_after_dropna` | covered |
| `s4_ids_before_drop` | metric | `step_4.py:110` | `test_step_4_metrics.py` | `test_ids_before_and_after_drop` | covered |
| `s4_ids_after_drop` | metric | `step_4.py:111` | `test_step_4_metrics.py` | `test_ids_before_and_after_drop` | covered |
| `step_4` error (empty) | error | `step_4.py:284` | `test_step_4_metrics.py` | `test_empty_file_error_tracked` | covered |
| `step_4` error (low rows) | error | `step_4.py:289` | `test_step_4_metrics.py` | `test_low_row_count_warning` | covered |
| `step_4` error (not found) | error | `step_4.py:260` | -- | not covered (file-not-found path) |
| `s4_best_val_loss` | metric | `step_4.py:200` | -- | not covered (requires LSTM training) |
| `s4_best_epoch` | metric | `step_4.py:201` | -- | not covered (requires LSTM training) |
| `s4_cards_with_price_vectors` | metric | `step_4.py:308` | -- | not covered (requires LSTM training) |
| `s4_duration` | metric | `step_4.py:313` | -- | not covered (timing only) |

---

## Step 5 (5 / 6 covered)

Strategy: **mock-and-intercept** -- creates tiny embedding parquet files, runs real
`run_step_5()` with real torch matmul on CPU (fast on small data).

| Widget ID | Type | Source | Test File | Test(s) | Status |
|-----------|------|--------|-----------|---------|--------|
| `s5_total_catalog_cards` | metric | `step_5.py:208` | `test_step_5_metrics.py` | `test_total_catalog_cards` | covered |
| `s5_cards_with_neighbors` | metric | `step_5.py:213` | `test_step_5_metrics.py` | `test_partial_coverage` | covered |
| `s5_catalog_coverage_pct` | metric | `step_5.py:218` | `test_step_5_metrics.py` | `test_coverage_percentage`, `test_partial_coverage` | covered |
| `s5_total_neighbor_pairs` | metric | `step_5.py:223` | `test_step_5_metrics.py` | `test_neighbor_pairs_count` | covered |
| `s5_db_size` | metric | `step_5.py:228` | `test_step_5_metrics.py` | `test_db_size_is_intersection` | covered |
| `s5_duration` | metric | `step_5.py:233` | -- | not covered (timing only) |

---

## Step 6 (0 / 3 covered)

Not tested. Step 6 has only trivial row-count and duration metrics. The step reads
two large parquet files and appends neighbor features -- heavy I/O with minimal
metric computation.

| Widget ID | Type | Source | Test File | Test(s) | Status |
|-----------|------|--------|-----------|---------|--------|
| `s6_historical_rows_output` | metric | `step_6.py:211` | -- | not covered (trivial row count) |
| `s6_today_rows_output` | metric | `step_6.py:216` | -- | not covered (trivial row count) |
| `s6_duration` | metric | `step_6.py:221` | -- | not covered (timing only) |

---

## Step 7 (0 / 3 covered)

Not tested. Step 7 uses `multiprocessing.Process` to train XGBoost models on
multiple GPUs. The validation MdAPE/MAPE metrics require a real trained model.

| Widget ID | Type | Source | Test File | Test(s) | Status |
|-----------|------|--------|-----------|---------|--------|
| `s7_validation_mdape` | metric | `step_7.py:104` | -- | not covered (requires XGBoost training) |
| `s7_validation_mape` | metric | `step_7.py:109` | -- | not covered (requires XGBoost training) |
| `s7_duration` | metric | `step_7.py:235` | -- | not covered (timing only) |

---

## Step 8 (12 / 14 covered)

Strategy: **replicate-and-verify** -- feeds known prediction arrays through the same
numpy min/max/mean/median/std computation that step 8 performs.

| Widget ID | Type | Source | Test File | Test(s) | Status |
|-----------|------|--------|-----------|---------|--------|
| `s8_predictions_generated` | metric | `step_8.py:170` | `test_step_8_metrics.py` | `test_basic_stats` | covered |
| `s8_prediction_min` | metric | `step_8.py:185` | `test_step_8_metrics.py` | `test_basic_stats`, `test_all_bounds_together` | covered |
| `s8_prediction_max` | metric | `step_8.py:190` | `test_step_8_metrics.py` | `test_basic_stats`, `test_all_bounds_together` | covered |
| `s8_prediction_mean` | metric | `step_8.py:195` | `test_step_8_metrics.py` | `test_basic_stats` | covered |
| `s8_prediction_median` | metric | `step_8.py:200` | `test_step_8_metrics.py` | `test_basic_stats` | covered |
| `s8_prediction_std` | metric | `step_8.py:205` | `test_step_8_metrics.py` | `test_basic_stats` | covered |
| `s8_negative_predictions_count` | metric | `step_8.py:210` | `test_step_8_metrics.py` | `test_negative_predictions_count`, `test_no_negative_predictions` | covered |
| `s8_prediction_lower_min` | metric | `step_8.py:229` | `test_step_8_metrics.py` | `test_lower_bound_stats`, `test_all_bounds_together` | covered |
| `s8_prediction_lower_max` | metric | `step_8.py:234` | `test_step_8_metrics.py` | `test_lower_bound_stats`, `test_all_bounds_together` | covered |
| `s8_prediction_upper_min` | metric | `step_8.py:242` | `test_step_8_metrics.py` | `test_upper_bound_stats`, `test_all_bounds_together` | covered |
| `s8_prediction_upper_max` | metric | `step_8.py:247` | `test_step_8_metrics.py` | `test_upper_bound_stats`, `test_all_bounds_together` | covered |
| `step_8` error (collapse) | error | `step_8.py:218` | `test_step_8_metrics.py` | `test_collapse_detected_when_all_same`, `test_no_collapse_when_varied` | covered |
| `s8_duration` | metric | `step_8.py:253` | -- | not covered (timing only) |

---

## Step 9 (5 / 6 covered)

Strategy: **mock-and-intercept** -- writes synthetic prediction and historical
parquet files, calls real `run_step_9()`.

| Widget ID | Type | Source | Test File | Test(s) | Status |
|-----------|------|--------|-----------|---------|--------|
| `s9_total_groups` | metric | `step_9.py:36` | `test_step_9_metrics.py` | `test_total_groups` | covered |
| `s9_monotonicity_violations_prediction` | metric | `step_9.py:51` | `test_step_9_metrics.py` | `test_violation_count_with_violations`, `test_no_violations_when_monotonic` | covered |
| `s9_post_sort_violations` | metric | `step_9.py:72` | `test_step_9_metrics.py` | `test_post_sort_violations_always_zero` | covered |
| `s9_qa_outliers` | metric | `step_9.py:161` | `test_step_9_metrics.py` | `test_outlier_count_matches_planted`, `test_zero_outliers_when_prices_match` | covered |
| `s9_qa_outlier_pct` | metric | `step_9.py:166` | `test_step_9_metrics.py` | `test_outlier_percentage` | covered |
| `s9_duration` | metric | `step_9.py:171` | -- | not covered (timing only) |

---

## Step 10 (0 / 2 covered)

Not tested. Step 10 uploads predictions to MongoDB via `MONGO_URI_RW`.
Both metrics are trivial (upload count and duration).

| Widget ID | Type | Source | Test File | Test(s) | Status |
|-----------|------|--------|-----------|---------|--------|
| `s10_documents_uploaded` | metric | `step_10.py:67` | -- | not covered (requires MongoDB RW) |
| `s10_duration` | metric | `step_10.py:72` | -- | not covered (timing only) |

---

## How to Add a Test for a New Metric

### 1. Pick the right strategy

| Strategy | Used by | When to use |
|----------|---------|-------------|
| **Mock-and-intercept** | Steps 1, 2, 5, 9 | Step has manageable I/O (1-3 mock targets). Call real `run_step_N()`, inspect tracker. |
| **Replicate-and-verify** | Steps 3, 4, 8 | Step has heavy I/O or deep coupling. Copy the metric's pandas/numpy formula, run it on synthetic data, push to tracker, assert. |

### 2. Mock-and-intercept pattern (example: adding `s1_new_metric` to step 1)

```python
# In tests/test_step_1_metrics.py

def test_new_metric(self, fresh_tracker, tmp_path):
    # Build synthetic data with known properties
    ebay_df, _ = make_ebay_df(n_rows=10, n_before_cutoff=0)
    pwcc_df, _ = make_pwcc_df(n_rows=5, n_before_cutoff=0)

    # Run the real step with mocked I/O
    _run_step_1_with_mocks(ebay_df, pwcc_df, tmp_path)

    # Verify the metric
    tracker = get_tracker()
    w = find_widget(tracker, "s1_new_metric")
    assert w is not None
    assert w["value"] == expected_value
```

If the metric needs new synthetic data properties, add a parameter to the
factory in `tests/helpers/synthetic_data.py`.

### 3. Replicate-and-verify pattern (example: adding `s8_prediction_p90` to step 8)

```python
# In tests/test_step_8_metrics.py

# First, add the computation to _simulate_step8_metrics():
pred_p90 = float(np.percentile(preds_arr, 90))
tracker.add_metric(id="s8_prediction_p90", ...)

# Then add a test:
def test_p90(self, fresh_tracker):
    predictions = [10.0, 20.0, 30.0, 40.0, 50.0]
    _simulate_step8_metrics(fresh_tracker, predictions)
    w = find_widget(fresh_tracker, "s8_prediction_p90")
    expected = round(float(np.percentile(predictions, 90)), 4)
    assert w["value"] == expected
```

### 4. Update this file

Add a row to the appropriate step table with the widget ID, source location,
test file, test method name, and mark it `covered`.
