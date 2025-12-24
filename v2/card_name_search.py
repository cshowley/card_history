import pandas as pd
import numpy as np
import torch
import re
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def get_device():
    """Detect the best available device for Apple Silicon or fallback."""
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def format_time(seconds: float) -> str:
    """Format seconds into human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


class SemanticSearch:
    def __init__(self, model_name: str = 'BAAI/bge-m3'):
        """Initialize the semantic search with the specified embedding model."""
        self.device = get_device()
        self.model_name = model_name
        self.model = None
        self.df = None
        self.embeddings = None
        self.column_name = None

    def _load_model(self):
        """Lazy load the model only when needed."""
        if self.model is None:
            print(f"Using device: {self.device}")
            print(f"Loading model: {self.model_name}...")
            start = time.time()
            self.model = SentenceTransformer(self.model_name, device=self.device)
            print(f"Model loaded in {format_time(time.time() - start)}")

    def _remove_grade_from_query(self, query: str) -> str:
        """
        Remove grading company names and grades from query string.
        
        Handles:
        - PSA 9, PSA 10
        - CGC 9.5, CGC 10 Pristine, CGC 9.5 Gem Mint
        - BGS 9.5, BGS 10 Pristine, BGS 9.5 Gem Mint
        - SGC 9.5, SGC 10
        
        Examples:
            "2022 #122 Duraludon V PSA 9 Play! Prize Pack" 
                -> "2022 #122 Duraludon V Play! Prize Pack"
            "2023 #036 Clefable CGC 10 Pristine Japanese" 
                -> "2023 #036 Clefable Japanese"
            "2017 #072 Looker CGC 9.5 Ultra Moon" 
                -> "2017 #072 Looker Ultra Moon"
        
        Args:
            query: The search query string
            
        Returns:
            Query string with grading info removed
        """
        # Grade labels that may follow the numeric grade (case insensitive)
        grade_labels = [
            r'Pristine',
            r'Gem[\s\-]?Mint',
            r'Mint\+?',
            r'NM[\s\-/]?MT\+?',
            r'NM\+?',
            r'VF[\s\-/]?NM\+?',
            r'VF\+?',
            r'F[\s\-/]?VF',
            r'Fine\+?',
            r'VG[\s\-/]?F',
            r'VG\+?',
            r'G[\s\-/]?VG',
            r'Good\+?',
            r'Fair',
            r'Poor',
            r'Authentic',
        ]
        
        # Build the grade label pattern
        label_pattern = '|'.join(grade_labels)
        
        # Pattern: (PSA|CGC|BGS|SGC) + space + number (with optional decimal) + optional grade label
        # Number can be 1-10 with optional decimal (e.g., 9, 9.5, 10)
        pattern = rf'\b(PSA|CGC|BGS|SGC)\s+\d+(\.\d+)?(\s+({label_pattern}))?\b'
        
        # Remove matches (case insensitive)
        cleaned = re.sub(pattern, '', query, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned

    def _extract_filters(self, query: str) -> dict:
        """
        Extract year and card number from query string for filtering.
        
        Args:
            query: The search query string
            
        Returns:
            Dict with 'year' and 'card_number' keys (values may be None)
        """
        filters = {
            'year': None,
            'card_number': None
        }
        
        # Match 4-digit year (2000-2099)
        year_match = re.search(r'\b(20\d{2})\b', query)
        if year_match:
            filters['year'] = year_match.group(1)
        
        # Match card number - try several patterns
        number_match = re.search(r'#\s*(\d+)', query)
        
        if not number_match:
            number_match = re.search(r'\b(\d+)\s*/\s*\d+', query)
        
        if not number_match:
            all_numbers = re.findall(r'\b(\d+)\b', query)
            for num in all_numbers:
                if num == filters['year']:
                    continue
                if len(num) <= 2 and int(num) <= 10:
                    grade_pattern = rf'(?:PSA|BGS|CGC|SGC)\s*{num}\b'
                    if re.search(grade_pattern, query, re.IGNORECASE):
                        continue
                number_match = type('Match', (), {'group': lambda self, x: num})()
                break
        
        if number_match:
            filters['card_number'] = number_match.group(1)
        
        return filters

    def _result_matches_filters(
        self, 
        result_text: str, 
        filters: dict,
        require_all: bool = True
    ) -> bool:
        """
        Check if result text contains required year and card number.
        """
        year = filters.get('year')
        card_number = filters.get('card_number')
        
        matches = []
        
        if year:
            year_found = bool(re.search(rf'\b{re.escape(year)}\b', result_text))
            matches.append(year_found)
        
        if card_number:
            patterns = [
                rf'#\s*{re.escape(card_number)}\b',
                rf'\b{re.escape(card_number)}\s*/\s*\d+',
                rf'\b{re.escape(card_number)}\b',
            ]
            number_found = any(re.search(p, result_text) for p in patterns)
            matches.append(number_found)
        
        if not matches:
            return True
        
        if require_all:
            return all(matches)
        else:
            return any(matches)

    def _result_matches_parallel(
        self, 
        result_row: pd.Series, 
        parallel_column: str = 'PARALLEL'
    ) -> bool:
        """
        Check if the PARALLEL field value is contained in the DETAILS field.
        Exception: "Base" is the implicit default and doesn't need to appear in DETAILS.
        """
        parallel = result_row.get(parallel_column, None)
        
        if parallel is None or pd.isna(parallel) or str(parallel).strip() == '':
            return True
        
        parallel = str(parallel).strip()
        
        if parallel.lower() == 'base':
            return True
        
        details = str(result_row.get(self.column_name, '')).strip()
        
        if re.search(re.escape(parallel), details, re.IGNORECASE):
            return True
        
        return False

    def load_and_embed_csv(
        self, 
        csv_path: str, 
        column_name: str = 'DETAILS',
        batch_size: int = 32
    ) -> pd.DataFrame:
        """
        Load a CSV file and embed the specified column.
        """
        self._load_model()
        self.column_name = column_name
        
        print(f"Loading CSV: {csv_path}...")
        self.df = pd.read_csv(csv_path)
        
        details = self.df[column_name].fillna('').astype(str).tolist()
        total_rows = len(details)
        
        print(f"Embedding {total_rows} rows...")
        start_time = time.time()
        
        self.embeddings = self.model.encode(
            details,
            normalize_embeddings=True,
            convert_to_tensor=True,
            show_progress_bar=True,
            device=self.device,
            batch_size=batch_size
        )
        
        elapsed = time.time() - start_time
        rate = total_rows / elapsed
        
        print(f"\nEmbedding complete!")
        print(f"  Total time: {format_time(elapsed)}")
        print(f"  Rate: {rate:.1f} rows/sec")
        print(f"  Shape: {self.embeddings.shape}")
        print(f"  Device: {self.embeddings.device}")
        
        return self.df

    def save(self, save_dir: str):
        """
        Save embeddings and dataframe to disk for fast loading later.
        """
        if self.embeddings is None or self.df is None:
            raise ValueError("No data to save. Call load_and_embed_csv first.")
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving to {save_dir}...")
        start = time.time()
        
        embeddings_path = save_path / "embeddings.pt"
        torch.save(self.embeddings.cpu(), embeddings_path)
        
        df_path = save_path / "dataframe.parquet"
        self.df.to_parquet(df_path, index=False)
        
        metadata = {
            'model_name': self.model_name,
            'column_name': self.column_name,
            'num_rows': len(self.df),
            'embedding_dim': self.embeddings.shape[1]
        }
        metadata_path = save_path / "metadata.pt"
        torch.save(metadata, metadata_path)
        
        elapsed = time.time() - start
        
        total_size = sum(f.stat().st_size for f in save_path.glob("*"))
        size_mb = total_size / (1024 * 1024)
        
        print(f"Saved in {format_time(elapsed)}")
        print(f"  Total size: {size_mb:.1f} MB")
        print(f"  Embeddings: {embeddings_path}")
        print(f"  DataFrame: {df_path}")

    def load(self, save_dir: str) -> pd.DataFrame:
        """
        Load pre-computed embeddings and dataframe from disk.
        """
        save_path = Path(save_dir)
        
        if not save_path.exists():
            raise FileNotFoundError(f"Save directory not found: {save_dir}")
        
        print(f"Loading from {save_dir}...")
        start = time.time()
        
        metadata_path = save_path / "metadata.pt"
        if metadata_path.exists():
            metadata = torch.load(metadata_path)
            print(f"  Model: {metadata['model_name']}")
            print(f"  Rows: {metadata['num_rows']}")
            print(f"  Embedding dim: {metadata['embedding_dim']}")
            self.column_name = metadata.get('column_name', 'DETAILS')
        
        embeddings_path = save_path / "embeddings.pt"
        with tqdm(total=1, desc="Loading embeddings", unit="file") as pbar:
            self.embeddings = torch.load(embeddings_path).to(self.device)
            pbar.update(1)
        
        df_path = save_path / "dataframe.parquet"
        with tqdm(total=1, desc="Loading dataframe", unit="file") as pbar:
            self.df = pd.read_parquet(df_path)
            pbar.update(1)
        
        elapsed = time.time() - start
        print(f"\nLoaded in {format_time(elapsed)}")
        print(f"  Embeddings shape: {self.embeddings.shape}")
        print(f"  Embeddings device: {self.embeddings.device}")
        
        return self.df

    def load_or_create(
        self, 
        csv_path: str, 
        cache_dir: str, 
        column_name: str = 'DETAILS',
        force_rebuild: bool = False,
        batch_size: int = 32
    ) -> pd.DataFrame:
        """
        Load from cache if available, otherwise create embeddings and save.
        """
        cache_path = Path(cache_dir)
        embeddings_exist = (cache_path / "embeddings.pt").exists()
        
        if embeddings_exist and not force_rebuild:
            print("Cache found! Loading from disk...")
            return self.load(cache_dir)
        else:
            if force_rebuild:
                print("Force rebuild requested...")
            else:
                print("No cache found, building embeddings...")
            
            self.load_and_embed_csv(csv_path, column_name, batch_size)
            self.save(cache_dir)
            return self.df

    def search(self, query: str, k: int = 5) -> np.ndarray:
        """
        Search for the k nearest neighbors to the query string.
        """
        indices, _ = self.search_with_scores(
            query, 
            k, 
            min_score=0.0,
            filter_by_year_and_number=False,
            filter_by_parallel=False,
            remove_grades=True
        )
        return indices

    def search_with_scores(
        self, 
        query: str, 
        k: int = 5,
        min_score: float = 0.8,
        filter_by_year_and_number: bool = True,
        filter_by_parallel: bool = True,
        remove_grades: bool = True,
        year: str = None,
        card_number: str = None,
        parallel_column: str = 'PARALLEL',
        search_multiplier: int = 50,
        verbose: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Search and return both indices and similarity scores.
        Applies multiple filters: year/card number, minimum score, and parallel matching.
        
        Args:
            query: The search string
            k: Number of nearest neighbors to return
            min_score: Minimum similarity score threshold (0.0 to 1.0)
            filter_by_year_and_number: If True, extract year/card number from query
                                       and filter results that don't contain both
            filter_by_parallel: If True, check that PARALLEL field value appears in DETAILS
                               (except for "Base" which is implicit)
            remove_grades: If True, remove PSA/CGC/BGS/SGC grades from query before
                          similarity search (e.g., "PSA 9", "CGC 10 Pristine")
            year: Explicitly specify year to filter on (overrides auto-extraction)
            card_number: Explicitly specify card number to filter on
            parallel_column: Name of the parallel column in dataframe
            search_multiplier: Fetch this many times more results before filtering
            verbose: If True, print filter information
            
        Returns:
            Tuple of (indices, similarity_scores) - may return fewer than k results
            if not enough results meet all filter criteria
        """
        if self.embeddings is None:
            raise ValueError("No embeddings built. Call load_and_embed_csv or load first.")
        
        self._load_model()
        
        # Extract filters from ORIGINAL query (before removing grades)
        filters = {'year': None, 'card_number': None}
        
        if filter_by_year_and_number:
            filters = self._extract_filters(query)
        
        if year is not None:
            filters['year'] = str(year)
        if card_number is not None:
            filters['card_number'] = str(card_number)
        
        # Remove grades from query for similarity search
        search_query = query
        if remove_grades:
            search_query = self._remove_grade_from_query(query)
        
        if verbose:
            filter_parts = []
            if filters['year']:
                filter_parts.append(f"Year: {filters['year']}")
            if filters['card_number']:
                filter_parts.append(f"Card #: {filters['card_number']}")
            filter_parts.append(f"Min score: {min_score}")
            if filter_by_parallel:
                filter_parts.append("Parallel: enabled")
            if remove_grades:
                filter_parts.append("Grades: removed")
            print(f"  Filters - {', '.join(filter_parts)}")
            
            if remove_grades and search_query != query:
                print(f"  Original query: {query}")
                print(f"  Search query:   {search_query}")
        
        # Determine how many results to fetch initially
        has_year_card_filters = filters['year'] is not None or filters['card_number'] is not None
        fetch_k = k * search_multiplier if (has_year_card_filters or filter_by_parallel) else k * 10
        fetch_k = min(fetch_k, len(self.embeddings))
        
        # Embed the CLEANED query (without grades)
        query_embedding = self.model.encode(
            search_query,
            normalize_embeddings=True,
            convert_to_tensor=True,
            device=self.device
        )
        
        # Compute similarities and get top results
        similarities = torch.mv(self.embeddings, query_embedding)
        top_k = torch.topk(similarities, k=fetch_k)
        
        indices = top_k.indices.cpu().numpy()
        scores = top_k.values.cpu().numpy()
        
        # Apply filtering
        filtered_indices = []
        filtered_scores = []
        checked_count = 0
        below_threshold_count = 0
        year_card_mismatch_count = 0
        parallel_mismatch_count = 0
        
        for idx, score in zip(indices, scores):
            checked_count += 1
            
            if score < min_score:
                below_threshold_count += 1
                break
            
            result_row = self.df.iloc[idx]
            result_text = str(result_row[self.column_name])
            
            if has_year_card_filters:
                if not self._result_matches_filters(result_text, filters):
                    year_card_mismatch_count += 1
                    continue
            
            if filter_by_parallel:
                if not self._result_matches_parallel(result_row, parallel_column):
                    parallel_mismatch_count += 1
                    if verbose and parallel_mismatch_count <= 3:
                        parallel_val = result_row.get(parallel_column, 'N/A')
                        print(f"    Parallel mismatch: '{parallel_val}' not found in '{result_text[:80]}...'")
                    continue
            
            filtered_indices.append(idx)
            filtered_scores.append(score)
            
            if len(filtered_indices) >= k:
                break
        
        if verbose:
            print(f"  Checked {checked_count} results: {len(filtered_indices)} passed, "
                  f"{year_card_mismatch_count} failed year/card#, "
                  f"{parallel_mismatch_count} failed parallel, "
                  f"{below_threshold_count} below score threshold")
        
        if len(filtered_indices) == 0:
            if verbose:
                print(f"  Warning: No results matched all criteria.")
            return np.array([]), np.array([])
        
        return np.array(filtered_indices), np.array(filtered_scores)

    def batch_search(
        self, 
        queries: list[str], 
        k: int = 5,
        min_score: float = 0.8,
        filter_by_year_and_number: bool = True,
        filter_by_parallel: bool = True,
        remove_grades: bool = True,
        parallel_column: str = 'PARALLEL',
        show_progress: bool = True
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Search for multiple queries at once.
        """
        if self.embeddings is None:
            raise ValueError("No embeddings built. Call load_and_embed_csv or load first.")
        
        self._load_model()
        
        results = []
        iterator = tqdm(queries, desc="Searching", unit="query") if show_progress else queries
        
        for query in iterator:
            indices, scores = self.search_with_scores(
                query, 
                k=k,
                min_score=min_score,
                filter_by_year_and_number=filter_by_year_and_number,
                filter_by_parallel=filter_by_parallel,
                remove_grades=remove_grades,
                parallel_column=parallel_column,
                verbose=False
            )
            results.append((indices, scores))
        
        return results


# Example usage
if __name__ == "__main__":
    searcher = SemanticSearch(model_name='BAAI/bge-m3')
    
    df = searcher.load_or_create(
        csv_path="gemrate_data.csv",
        cache_dir="./embeddings_cache",
        column_name="DETAILS",
        force_rebuild=False
    )
    
    # Test grade removal
    print("\n" + "="*70)
    print("GRADE REMOVAL EXAMPLES")
    print("="*70)
    
    test_queries = [
        "2022 #122 Duraludon V PSA 9 Play! Prize Pack Pokemon",
        "2023 #036 Clefable Master Ball Reverse Holo CGC 10 Pristine Japanese Sv2a- 151 Pokemon",
        "2017 #072 Looker CGC 9.5 Ultra Moon - SM5M - Japanese Pokemon",
        "2021 #25 Pikachu BGS 9.5 Gem Mint Celebrations Pokemon",
        "2020 #4 Charizard V SGC 10 Champion's Path Pokemon",
    ]
    
    print("\nGrade removal test:")
    for query in test_queries:
        cleaned = searcher._remove_grade_from_query(query)
        print(f"\n  Original: {query}")
        print(f"  Cleaned:  {cleaned}")
    
    # Search examples
    print("\n" + "="*70)
    print("SEARCH WITH GRADE REMOVAL")
    print("="*70)
    
    query = "2022 #122 Duraludon V PSA 9 Play! Prize Pack Pokemon"
    print(f"\nQuery: {query}")
    
    indices, scores = searcher.search_with_scores(
        query, 
        k=5,
        min_score=0.8,
        filter_by_year_and_number=True,
        filter_by_parallel=True,
        remove_grades=True
    )
    
    if len(indices) > 0:
        print(f"\nTop {len(indices)} matches:")
        for i, (idx, score) in enumerate(zip(indices, scores), 1):
            row = df.iloc[idx]
            parallel = row.get('PARALLEL', 'N/A')
            print(f"  {i}. [{score:.4f}] [Parallel: {parallel}] {row['DETAILS']}")
    else:
        print("\nNo matches found.")
    
    # Compare with and without grade removal
    print("\n" + "="*70)
    print("COMPARISON: WITH vs WITHOUT GRADE REMOVAL")
    print("="*70)
    
    query2 = "2023 #036 Clefable Master Ball Reverse Holo CGC 10 Pristine Japanese Sv2a- 151 Pokemon"
    print(f"\nQuery: {query2}")
    
    print("\n--- WITH grade removal ---")
    indices_with, scores_with = searcher.search_with_scores(
        query2,
        k=3,
        min_score=0.7,
        remove_grades=True
    )
    
    if len(indices_with) > 0:
        for i, (idx, score) in enumerate(zip(indices_with, scores_with), 1):
            print(f"  {i}. [{score:.4f}] {df.iloc[idx]['DETAILS'][:80]}...")
    
    print("\n--- WITHOUT grade removal ---")
    indices_without, scores_without = searcher.search_with_scores(
        query2,
        k=3,
        min_score=0.7,
        remove_grades=False
    )
    
    if len(indices_without) > 0:
        for i, (idx, score) in enumerate(zip(indices_without, scores_without), 1):
            print(f"  {i}. [{score:.4f}] {df.iloc[idx]['DETAILS'][:80]}...")