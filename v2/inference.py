"""
Inference module for card embedding nearest-neighbor search.

Usage:
    from inference import CardEmbeddingIndex

    index = CardEmbeddingIndex()
    neighbors = index.find_nearest("12345678", n=10)
    # Returns: [("spec_id_1", 0.95), ("spec_id_2", 0.93), ...]
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Optional

# =========================
# CARD EMBEDDING INDEX
# =========================


class CardEmbeddingIndex:
    """
    Loads precomputed card embeddings and provides nearest-neighbor search.
    """

    def __init__(
        self,
        embeddings_path: str = "card_vectors_768.npy",
        spec_ids_path: str = "spec_ids.npy",
        sales_spec_ids_path: str = "sales_spec_ids.npy",
        spec_id_grades_path: str = "spec_id_grades.pkl",
        latest_prices_by_grade_path: str = "latest_prices_by_grade.pkl",
        metadata_path: Optional[str] = None,
    ):
        """
        Initialize the embedding index.

        Args:
            embeddings_path: Path to .npy file with shape (n_cards, 768)
            spec_ids_path: Path to .npy file with spec_id ordering (same order as embeddings)
            sales_spec_ids_path: Path to .npy file with spec_ids that have sales history
            spec_id_grades_path: Path to pickle file with spec_id -> set of grades mapping
            latest_prices_by_grade_path: Path to pickle file with spec_id -> {grade: price}
            metadata_path: Path to card metadata CSV (optional, for enriched results)
        """
        base_dir = os.path.dirname(__file__)
        embeddings_path = os.path.join(base_dir, embeddings_path)
        spec_ids_path = os.path.join(base_dir, spec_ids_path)
        sales_spec_ids_path = os.path.join(base_dir, sales_spec_ids_path)
        spec_id_grades_path = os.path.join(base_dir, spec_id_grades_path)
        latest_prices_by_grade_path = os.path.join(
            base_dir, latest_prices_by_grade_path
        )

        print(f"Loading embeddings from {embeddings_path}...")
        self.embeddings = np.load(embeddings_path).astype(np.float32)
        print(
            f"  → Loaded {self.embeddings.shape[0]} embeddings of dim {self.embeddings.shape[1]}"
        )

        # L2 normalize for cosine similarity via dot product
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # avoid division by zero
        self.embeddings_normalized = self.embeddings / norms

        # Load spec_ids
        print(f"Loading spec_ids from {spec_ids_path}...")
        if spec_ids_path.endswith(".npy"):
            self.spec_ids = np.load(spec_ids_path).astype(str)
        else:
            self.spec_ids = pd.read_csv(spec_ids_path).iloc[:, 0].astype(str).values

        if len(self.spec_ids) != len(self.embeddings):
            raise ValueError(
                f"Mismatch: {len(self.spec_ids)} spec_ids vs {len(self.embeddings)} embeddings"
            )

        # Build spec_id -> index mapping
        self.spec_id_to_idx = {sid: i for i, sid in enumerate(self.spec_ids)}

        # Load sales_spec_ids (spec_ids that have sales history)
        self.sales_spec_ids_set = set()
        if os.path.exists(sales_spec_ids_path):
            print(f"Loading sales_spec_ids from {sales_spec_ids_path}...")
            sales_spec_ids = np.load(sales_spec_ids_path).astype(str)
            self.sales_spec_ids_set = set(sales_spec_ids)
            print(f"  → {len(self.sales_spec_ids_set)} spec_ids have sales history")
        else:
            print(f"Warning: {sales_spec_ids_path} not found, sales filtering disabled")

        # Load spec_id_grades mapping (spec_id -> set of grades)
        self.spec_id_grades = {}
        if os.path.exists(spec_id_grades_path):
            print(f"Loading spec_id_grades from {spec_id_grades_path}...")
            with open(spec_id_grades_path, "rb") as f:
                self.spec_id_grades = pickle.load(f)
            print(f"  → {len(self.spec_id_grades)} spec_ids have grade data")
        else:
            print(f"Warning: {spec_id_grades_path} not found, grade filtering disabled")

        # Load latest prices by grade mapping (spec_id -> {grade: price})
        self.latest_prices_by_grade = {}
        if os.path.exists(latest_prices_by_grade_path):
            print(
                f"Loading latest_prices_by_grade from {latest_prices_by_grade_path}..."
            )
            with open(latest_prices_by_grade_path, "rb") as f:
                self.latest_prices_by_grade = pickle.load(f)
            print(
                f"  → {len(self.latest_prices_by_grade)} spec_ids have grade-specific prices"
            )
        else:
            print(
                f"Warning: {latest_prices_by_grade_path} not found, price lookup disabled"
            )

        # Optionally load full metadata for enriched results
        self.metadata = None
        if metadata_path is not None:
            metadata_path = os.path.join(base_dir, metadata_path)
            if os.path.exists(metadata_path):
                self.metadata = pd.read_csv(metadata_path)

        print(f"Index ready: {len(self.spec_ids)} cards")

    def get_card_name(self, spec_id: str) -> Optional[str]:
        """Get the card name for a given spec_id."""
        spec_id = str(spec_id)
        if self.metadata is None:
            print("Warning: No metadata loaded, cannot look up card names")
            return None

        # Find the spec_id column
        spec_col = None
        for col in ["SPECID", "spec_id", "specid"]:
            if col in self.metadata.columns:
                spec_col = col
                break

        if spec_col is None:
            print("Warning: No spec_id column found in metadata")
            return None

        # Find the NAME column
        name_col = None
        for col in ["NAME", "name", "Name"]:
            if col in self.metadata.columns:
                name_col = col
                break

        if name_col is None:
            print("Warning: No NAME column found in metadata")
            return None

        match = self.metadata[self.metadata[spec_col].astype(str) == spec_id]
        if len(match) == 0:
            return None
        return match.iloc[0][name_col]

    def get_card_info(self, spec_id: str) -> Optional[dict]:
        """Get all metadata for a given spec_id."""
        spec_id = str(spec_id)
        if self.metadata is None:
            print("Warning: No metadata loaded")
            return None

        # Find the spec_id column
        spec_col = None
        for col in ["SPECID", "spec_id", "specid"]:
            if col in self.metadata.columns:
                spec_col = col
                break

        if spec_col is None:
            return None

        match = self.metadata[self.metadata[spec_col].astype(str) == spec_id]
        if len(match) == 0:
            return None
        return match.iloc[0].to_dict()

    def get_embedding(self, spec_id: str) -> Optional[np.ndarray]:
        """Get the embedding vector for a spec_id."""
        spec_id = str(spec_id)
        if spec_id not in self.spec_id_to_idx:
            return None
        idx = self.spec_id_to_idx[spec_id]
        return self.embeddings[idx]

    def find_nearest(
        self,
        spec_id: str,
        n: int = 10,
        include_self: bool = False,
        has_sales_only: bool = False,
        grade: Optional[int] = None,
    ) -> list[dict]:
        """
        Find the N nearest neighbors to a given spec_id.

        Args:
            spec_id: The card spec_id to search from
            n: Number of neighbors to return
            include_self: Whether to include the query card itself
            has_sales_only: If True, only return neighbors that have sales history
            grade: If provided, only return neighbors with sales at this grade

        Returns:
            List of dicts with keys: spec_id, similarity, price, grade
        """
        spec_id = str(spec_id)

        if spec_id not in self.spec_id_to_idx:
            raise ValueError(f"spec_id '{spec_id}' not found in index")

        query_idx = self.spec_id_to_idx[spec_id]
        query_vec = self.embeddings_normalized[query_idx]

        # Compute cosine similarity with all embeddings
        # Since both are normalized, dot product = cosine similarity
        similarities = self.embeddings_normalized @ query_vec

        # Sort all indices by similarity (descending)
        sorted_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in sorted_indices:
            neighbor_id = self.spec_ids[idx]

            # Skip self if not included
            if not include_self and neighbor_id == spec_id:
                continue

            # Skip if filtering by sales and neighbor has no sales
            if has_sales_only and neighbor_id not in self.sales_spec_ids_set:
                continue

            # Skip if filtering by grade and neighbor doesn't have that grade
            if grade is not None:
                neighbor_grades = self.spec_id_grades.get(neighbor_id, set())
                if grade not in neighbor_grades:
                    continue

            # Get price based on grade filter
            if grade is not None:
                # Get price for specific grade
                grade_prices = self.latest_prices_by_grade.get(neighbor_id, {})
                price = grade_prices.get(grade, None)
            else:
                # No grade filter - get any available price (pick lowest grade's price)
                grade_prices = self.latest_prices_by_grade.get(neighbor_id, {})
                if grade_prices:
                    # Use the price from the lowest available grade
                    min_grade = min(grade_prices.keys())
                    price = grade_prices[min_grade]
                else:
                    price = None

            results.append(
                {
                    "spec_id": neighbor_id,
                    "similarity": float(similarities[idx]),
                    "price": price,
                    "grade": grade,
                }
            )
            if len(results) >= n:
                break

        return results

    def find_nearest_with_metadata(
        self,
        spec_id: str,
        n: int = 10,
        include_self: bool = False,
        has_sales_only: bool = False,
    ) -> pd.DataFrame:
        """
        Find nearest neighbors and return enriched results with metadata.

        Args:
            spec_id: The card spec_id to search from
            n: Number of neighbors to return
            include_self: Whether to include the query card itself
            has_sales_only: If True, only return neighbors that have sales history

        Returns:
            DataFrame with columns: spec_id, similarity, price, and all metadata columns
        """
        neighbors = self.find_nearest(spec_id, n, include_self, has_sales_only)

        # Build result DataFrame
        result_data = []
        for neighbor in neighbors:
            neighbor_id = neighbor["spec_id"]
            row = {
                "spec_id": neighbor_id,
                "similarity": neighbor["similarity"],
                "price": neighbor["price"],
            }

            if self.metadata is not None:
                # Find metadata for this neighbor
                for col in ["SPECID", "spec_id", "specid"]:
                    if col in self.metadata.columns:
                        match = self.metadata[
                            self.metadata[col].astype(str) == neighbor_id
                        ]
                        if len(match) > 0:
                            for c in self.metadata.columns:
                                if c not in ["SPECID", "spec_id", "specid"]:
                                    row[c] = match.iloc[0][c]
                        break

            result_data.append(row)

        return pd.DataFrame(result_data)

    def batch_find_nearest(
        self,
        spec_ids: list[str],
        n: int = 10,
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Find nearest neighbors for multiple spec_ids efficiently.

        Args:
            spec_ids: List of spec_ids to search
            n: Number of neighbors per query

        Returns:
            Dict mapping each query spec_id to its list of neighbors
        """
        results = {}
        for sid in spec_ids:
            try:
                results[sid] = self.find_nearest(sid, n)
            except ValueError:
                results[sid] = []
        return results


# =========================
# EXAMPLE USAGE
# =========================

if __name__ == "__main__":
    # Load the index
    index = CardEmbeddingIndex()

    # Example: Find nearest neighbors for a sample card
    sample_spec_ids = list(index.spec_id_to_idx.keys())[:2]

    for spec_id in sample_spec_ids:
        print(f"\n{'='*60}")
        print(f"Query: spec_id = {spec_id}")
        print(f"{'='*60}")

        # All neighbors (regardless of sales history)
        print(f"\nTop 5 nearest neighbors (all, no grade filter):")
        neighbors = index.find_nearest(spec_id, n=5, has_sales_only=False)
        for i, neighbor in enumerate(neighbors, 1):
            price_str = f"${neighbor['price']:.2f}" if neighbor["price"] else "N/A"
            grade_str = neighbor["grade"] if neighbor["grade"] is not None else "N/A"
            print(
                f"  {i}. spec_id={neighbor['spec_id']}, "
                f"sim={neighbor['similarity']:.4f}, price={price_str}, grade={grade_str}"
            )

        # Only neighbors with sales at grade 9
        print(f"\nTop 5 nearest neighbors (grade=9 only):")
        neighbors = index.find_nearest(spec_id, n=5, has_sales_only=True, grade=9)
        for i, neighbor in enumerate(neighbors, 1):
            price_str = f"${neighbor['price']:.2f}" if neighbor["price"] else "N/A"
            print(
                f"  {i}. spec_id={neighbor['spec_id']}, "
                f"sim={neighbor['similarity']:.4f}, price={price_str}, grade={neighbor['grade']}"
            )

        # Only neighbors with sales at grade 8
        print(f"\nTop 5 nearest neighbors (grade=8 only):")
        neighbors = index.find_nearest(spec_id, n=5, has_sales_only=True, grade=8)
        for i, neighbor in enumerate(neighbors, 1):
            price_str = f"${neighbor['price']:.2f}" if neighbor["price"] else "N/A"
            print(
                f"  {i}. spec_id={neighbor['spec_id']}, "
                f"sim={neighbor['similarity']:.4f}, price={price_str}, grade={neighbor['grade']}"
            )
