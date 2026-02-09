"""
Sanity-check tests for Step 5 data integrity metrics.

Creates synthetic text_embeddings and price_embeddings parquet files,
runs the real run_step_5(), and verifies neighbor search coverage metrics.
"""

import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from data_integrity import get_tracker
from tests.conftest import find_widget


def _make_text_embeddings(path, ids, dim=16):
    """Write a synthetic text_embeddings.parquet.

    Args:
        ids: List of gemrate_id strings.
        dim: Embedding dimension.
    """
    rng = np.random.default_rng(42)
    vecs = rng.normal(size=(len(ids), dim))
    df = pd.DataFrame({
        "gemrate_id": ids,
        "embedding_vector": list(vecs),
    })
    df.to_parquet(str(path), index=False)
    return df


def _make_price_embeddings(path, ids, dim=8):
    """Write a synthetic price_embeddings.parquet.

    Args:
        ids: List of gemrate_id strings (index-based, no gemrate_id column).
        dim: Embedding dimension.
    """
    rng = np.random.default_rng(99)
    vecs = rng.normal(size=(len(ids), dim))
    df = pd.DataFrame(vecs, index=pd.Index(ids, name="gemrate_id"))
    df.to_parquet(str(path))
    return df


def _run_step_5_with_mocks(tmp_path, text_ids, price_ids, n_neighbors=3):
    """Run step_5 with file paths pointed at synthetic data."""
    text_emb_path = tmp_path / "text_embeddings.parquet"
    price_emb_path = tmp_path / "price_embeddings.parquet"
    output_path = tmp_path / "neighbors.parquet"

    _make_text_embeddings(text_emb_path, text_ids)
    _make_price_embeddings(price_emb_path, price_ids)

    with (
        patch("step_5.constants.S2_OUTPUT_EMBEDDINGS_FILE", str(text_emb_path)),
        patch("step_5.constants.S4_OUTPUT_PRICE_VECS_FILE", str(price_emb_path)),
        patch("step_5.constants.S5_OUTPUT_NEIGHBORS_FILE", str(output_path)),
        patch("step_5.constants.S5_N_NEIGHBORS_PREPARE", n_neighbors),
        patch("step_5.constants.DEVICE", "cpu"),
    ):
        from step_5 import run_step_5
        run_step_5()

    return output_path


class TestStep5CoverageMetrics:
    """Test neighbor search coverage and count metrics."""

    def test_total_catalog_cards(self, fresh_tracker, tmp_path):
        """s5_total_catalog_cards should equal the number of query IDs (all text embeddings)."""
        text_ids = [f"GEM_{i:04d}" for i in range(10)]
        price_ids = [f"GEM_{i:04d}" for i in range(6)]  # only 6 have price vecs

        _run_step_5_with_mocks(tmp_path, text_ids, price_ids, n_neighbors=3)

        tracker = get_tracker()
        w = find_widget(tracker, "s5_total_catalog_cards")
        assert w is not None
        assert w["value"] == 10  # all text embedding IDs are the query set

    def test_db_size_is_intersection(self, fresh_tracker, tmp_path):
        """s5_db_size should be the intersection of text and price embedding IDs."""
        text_ids = [f"GEM_{i:04d}" for i in range(10)]
        price_ids = [f"GEM_{i:04d}" for i in range(4, 8)]  # overlap: 4,5,6,7

        _run_step_5_with_mocks(tmp_path, text_ids, price_ids, n_neighbors=2)

        tracker = get_tracker()
        w = find_widget(tracker, "s5_db_size")
        assert w is not None
        assert w["value"] == 4  # intersection size

    def test_coverage_percentage(self, fresh_tracker, tmp_path):
        """Coverage should be (cards with neighbors / total catalog cards) * 100."""
        # All 8 text IDs can query against 8 price IDs (full overlap)
        ids = [f"GEM_{i:04d}" for i in range(8)]

        _run_step_5_with_mocks(tmp_path, ids, ids, n_neighbors=3)

        tracker = get_tracker()
        w = find_widget(tracker, "s5_catalog_coverage_pct")
        assert w is not None
        assert w["value"] == 100.0

    def test_partial_coverage(self, fresh_tracker, tmp_path):
        """Cards without price embeddings have no potential neighbors in the DB."""
        text_ids = [f"GEM_{i:04d}" for i in range(10)]
        # Only GEM_0000..GEM_0004 have price vecs (DB size = 5)
        # Those 5 IDs CAN find neighbors among themselves.
        # The other 5 text-only IDs search against the DB but have no price
        # component -- they still get text-only similarity matches from the DB.
        price_ids = [f"GEM_{i:04d}" for i in range(5)]

        _run_step_5_with_mocks(tmp_path, text_ids, price_ids, n_neighbors=2)

        tracker = get_tracker()
        w_total = find_widget(tracker, "s5_total_catalog_cards")
        w_with = find_widget(tracker, "s5_cards_with_neighbors")
        w_pct = find_widget(tracker, "s5_catalog_coverage_pct")

        assert w_total["value"] == 10
        # All 10 query IDs should still find neighbors (text similarity works for all)
        assert w_with["value"] > 0
        assert w_pct["value"] == round((w_with["value"] / 10) * 100, 2)

    def test_neighbor_pairs_count(self, fresh_tracker, tmp_path):
        """Total neighbor pairs should be > 0 when there is data."""
        ids = [f"GEM_{i:04d}" for i in range(6)]

        _run_step_5_with_mocks(tmp_path, ids, ids, n_neighbors=3)

        tracker = get_tracker()
        w = find_widget(tracker, "s5_total_neighbor_pairs")
        assert w is not None
        # Each of 6 IDs gets up to 3 neighbors (excluding self), so max 6*3=18
        # but at least some pairs should exist
        assert w["value"] > 0
        assert w["value"] <= 6 * 3
