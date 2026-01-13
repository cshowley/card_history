"""Cache for market index data."""

import time
from datetime import datetime
from typing import Optional

import requests

from api.config import get_settings


class IndexCache:
    """Caches market index data with TTL-based refresh."""

    def __init__(self):
        self._index_data: dict[str, float] = {}  # date_str -> index_value
        self._last_refresh: Optional[float] = None
        self._latest_index: Optional[float] = None
        self._index_change_1d: Optional[float] = None
        self._index_change_1w: Optional[float] = None
        self._index_ema_12: Optional[float] = None
        self._index_ema_26: Optional[float] = None

    def refresh(self) -> bool:
        """Fetch fresh index data from API."""
        settings = get_settings()

        try:
            response = requests.get(settings.index_api_url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if not data:
                return False

            # Store data by date
            self._index_data = {}
            for item in data:
                date_str = item.get("date", "")[:10]  # YYYY-MM-DD
                value = item.get("value")
                if date_str and value is not None:
                    self._index_data[date_str] = float(value)

            # Calculate derived values
            if self._index_data:
                sorted_dates = sorted(self._index_data.keys())
                values = [self._index_data[d] for d in sorted_dates]

                # Latest value
                self._latest_index = values[-1] if values else None

                # 1-day change
                if len(values) >= 2:
                    self._index_change_1d = values[-1] - values[-2]
                else:
                    self._index_change_1d = None

                # 1-week change (7 days)
                if len(values) >= 8:
                    self._index_change_1w = values[-1] - values[-8]
                else:
                    self._index_change_1w = None

                # EMA calculations
                self._index_ema_12 = self._calculate_ema(values, 12)
                self._index_ema_26 = self._calculate_ema(values, 26)

            self._last_refresh = time.time()
            print(f"Index cache refreshed: {len(self._index_data)} data points")
            return True

        except Exception as e:
            print(f"Error refreshing index cache: {e}")
            return False

    def _calculate_ema(self, values: list[float], span: int) -> Optional[float]:
        """Calculate exponential moving average."""
        if len(values) < span:
            return values[-1] if values else None

        multiplier = 2 / (span + 1)
        ema = values[0]
        for value in values[1:]:
            ema = (value * multiplier) + (ema * (1 - multiplier))
        return ema

    def _should_refresh(self) -> bool:
        """Check if cache should be refreshed based on TTL."""
        if self._last_refresh is None:
            return True

        settings = get_settings()
        elapsed = time.time() - self._last_refresh
        return elapsed > settings.index_cache_ttl_seconds

    def get_index_features(self) -> dict:
        """Get all index-related features for model input."""
        if self._should_refresh():
            self.refresh()

        return {
            "index_value": self._latest_index,
            "index_change_1d": self._index_change_1d,
            "index_change_1w": self._index_change_1w,
            "index_ema_12": self._index_ema_12,
            "index_ema_26": self._index_ema_26,
        }

    def get_index_for_date(self, date: datetime) -> Optional[float]:
        """Get index value for a specific date."""
        if self._should_refresh():
            self.refresh()

        date_str = date.strftime("%Y-%m-%d")
        return self._index_data.get(date_str)


# Global instance
_index_cache: Optional[IndexCache] = None


def get_index_cache() -> IndexCache:
    """Get singleton index cache instance."""
    global _index_cache
    if _index_cache is None:
        _index_cache = IndexCache()
    return _index_cache
