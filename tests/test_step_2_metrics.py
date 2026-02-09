"""
Sanity-check tests for Step 2 data integrity metrics.

Mocks SentenceTransformer to avoid loading the real model, injects a
synthetic catalog CSV, then verifies text quality and embedding metrics.
"""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from data_integrity import get_tracker
from tests.conftest import find_widget
from tests.helpers.synthetic_data import make_catalog_csv


def _run_step_2_with_mocks(catalog_path, tmp_path, n_rows_after_dedup=5, embedding_dim=128):
    """Run step_2 with SentenceTransformer mocked."""
    mock_model = MagicMock()
    fake_embeddings = np.random.default_rng(42).normal(
        size=(n_rows_after_dedup, embedding_dim)
    )
    mock_model.encode.return_value = fake_embeddings

    output_parquet = str(tmp_path / "text_embeddings.parquet")

    with (
        patch("step_2.SentenceTransformer", return_value=mock_model),
        patch("step_2.constants.S2_INPUT_CATALOG_FILE", str(catalog_path)),
        patch("step_2.constants.S2_OUTPUT_EMBEDDINGS_FILE", output_parquet),
        patch("step_2.constants.S2_MODEL_NAME", "fake-model"),
        patch("step_2.constants.DEVICE", "cpu"),
    ):
        from step_2 import run_step_2
        run_step_2()

    return fake_embeddings


class TestStep2TextQuality:
    """Test text quality metrics against known catalog data."""

    def test_input_rows_after_dedup(self, fresh_tracker, tmp_path):
        catalog_path = tmp_path / "catalog.csv"
        expected = make_catalog_csv(catalog_path, n_rows=5, n_empty_text=0, n_duplicates=2)

        _run_step_2_with_mocks(catalog_path, tmp_path, n_rows_after_dedup=expected["s2_input_rows"])

        tracker = get_tracker()
        w = find_widget(tracker, "s2_input_rows")
        assert w is not None
        assert w["value"] == expected["s2_input_rows"]

    def test_empty_text_count(self, fresh_tracker, tmp_path):
        catalog_path = tmp_path / "catalog.csv"
        expected = make_catalog_csv(catalog_path, n_rows=8, n_empty_text=3, n_duplicates=0)

        _run_step_2_with_mocks(
            catalog_path, tmp_path,
            n_rows_after_dedup=expected["s2_input_rows"],
        )

        tracker = get_tracker()
        w = find_widget(tracker, "s2_empty_text_count")
        assert w is not None
        assert w["value"] == expected["s2_empty_text_count"]

    def test_avg_text_length_positive(self, fresh_tracker, tmp_path):
        catalog_path = tmp_path / "catalog.csv"
        make_catalog_csv(catalog_path, n_rows=5, n_empty_text=0, n_duplicates=0)

        _run_step_2_with_mocks(catalog_path, tmp_path, n_rows_after_dedup=5)

        tracker = get_tracker()
        w = find_widget(tracker, "s2_avg_text_length")
        assert w is not None
        assert w["value"] > 0


class TestStep2EmbeddingOutput:
    """Test embedding output shape metrics."""

    def test_cards_embedded(self, fresh_tracker, tmp_path):
        catalog_path = tmp_path / "catalog.csv"
        make_catalog_csv(catalog_path, n_rows=6, n_empty_text=0, n_duplicates=0)

        _run_step_2_with_mocks(catalog_path, tmp_path, n_rows_after_dedup=6)

        tracker = get_tracker()
        w = find_widget(tracker, "s2_cards_embedded")
        assert w is not None
        assert w["value"] == 6

    def test_embedding_dimension(self, fresh_tracker, tmp_path):
        catalog_path = tmp_path / "catalog.csv"
        make_catalog_csv(catalog_path, n_rows=4, n_empty_text=0, n_duplicates=0)

        _run_step_2_with_mocks(catalog_path, tmp_path, n_rows_after_dedup=4, embedding_dim=256)

        tracker = get_tracker()
        w = find_widget(tracker, "s2_embedding_dim")
        assert w is not None
        assert w["value"] == 256

    def test_embedding_failure_tracked(self, fresh_tracker, tmp_path):
        """When model.encode() raises, error and failure flag should be tracked."""
        catalog_path = tmp_path / "catalog.csv"
        make_catalog_csv(catalog_path, n_rows=3, n_empty_text=0, n_duplicates=0)

        mock_model = MagicMock()
        mock_model.encode.side_effect = RuntimeError("GPU OOM")

        output_parquet = str(tmp_path / "text_embeddings.parquet")

        with (
            patch("step_2.SentenceTransformer", return_value=mock_model),
            patch("step_2.constants.S2_INPUT_CATALOG_FILE", str(catalog_path)),
            patch("step_2.constants.S2_OUTPUT_EMBEDDINGS_FILE", output_parquet),
            patch("step_2.constants.S2_MODEL_NAME", "fake-model"),
            patch("step_2.constants.DEVICE", "cpu"),
        ):
            from step_2 import run_step_2
            with pytest.raises(RuntimeError, match="GPU OOM"):
                run_step_2()

        tracker = get_tracker()
        # Error should be recorded
        errors = tracker.get_data()["errors"]
        assert any("GPU OOM" in e for e in errors)

        # Failure flag should be set
        w = find_widget(tracker, "s2_embedding_failed")
        assert w is not None
        assert w["value"] is True
