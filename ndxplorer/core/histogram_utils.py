"""Histogram utilities for computation, caching, and management."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from .histograms import Histogram1D, Histogram2D, Histogram3D


@dataclass
class HistogramParams:
    """Parameters that uniquely determine histogram computation."""
    x_param: str
    y_param: str
    z_param: Optional[str] = None
    x_bins_1d: str = "50"
    y_bins_1d: str = "50"
    z_bins_1d: str = "50"
    x_bins_2d: str = "50"
    y_bins_2d: str = "50"
    use_weights: bool = False
    weight_param: str = ""
    z_enabled: bool = False
    mask_id: Optional[str] = None
    data_hash: Optional[str] = None
    normed_x: bool = False
    normed_y: bool = False
    normed_z: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for caching."""
        return {
            'x_param': self.x_param,
            'y_param': self.y_param,
            'z_param': self.z_param,
            'x_bins_1d': self.x_bins_1d,
            'y_bins_1d': self.y_bins_1d,
            'z_bins_1d': self.z_bins_1d,
            'x_bins_2d': self.x_bins_2d,
            'y_bins_2d': self.y_bins_2d,
            'use_weights': self.use_weights,
            'weight_param': self.weight_param,
            'z_enabled': self.z_enabled,
            'mask_id': self.mask_id,
            'data_hash': self.data_hash,
            'normed_x': self.normed_x,
            'normed_y': self.normed_y,
            'normed_z': self.normed_z
        }


class HistogramCache:
    """Simple, efficient histogram cache with LRU eviction."""

    def __init__(self, max_size_mb: float = 200.0):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._max_size_mb = max_size_mb
        self._current_size_mb = 0.0
        self._hits = 0
        self._misses = 0

    def _compute_cache_key(self, params: Dict[str, Any]) -> str:
        """Compute deterministic cache key from parameters."""
        key_data = {k: v for k, v in params.items() if v is not None}
        key_str = json.dumps(key_data, sort_keys=True, separators=(',', ':'))
        return hashlib.md5(key_str.encode()).hexdigest()

    def _estimate_size_mb(self, hist: Union[Histogram1D, Histogram2D, Histogram3D]) -> float:
        """Estimate memory usage in MB."""
        if isinstance(hist, Histogram3D):
            return (hist.H.nbytes + hist.x_edges.nbytes + hist.y_edges.nbytes + hist.z_edges.nbytes) / (1024 * 1024)
        elif isinstance(hist, Histogram2D):
            return (hist.H.nbytes + hist.x_edges.nbytes + hist.y_edges.nbytes) / (1024 * 1024)
        else:
            return (hist.counts.nbytes + hist.edges.nbytes) / (1024 * 1024)

    def get(self, params: Dict[str, Any], hist_type: str = '2d') -> Optional[Union[Histogram1D, Histogram2D, Histogram3D]]:
        """Get cached histogram."""
        key = self._compute_cache_key(params)
        cache_key = f"{hist_type}_{key}"

        if cache_key in self._cache:
            self._hits += 1
            return self._cache[cache_key]['histogram']

        self._misses += 1
        return None

    def put(self, params: Dict[str, Any], histogram: Union[Histogram1D, Histogram2D, Histogram3D]) -> None:
        """Store histogram in cache with LRU eviction."""
        if not histogram.validate():
            return

        hist_type = '3d' if isinstance(histogram, Histogram3D) else ('2d' if isinstance(histogram, Histogram2D) else '1d')
        key = self._compute_cache_key(params)
        cache_key = f"{hist_type}_{key}"

        size_mb = self._estimate_size_mb(histogram)
        if size_mb > self._max_size_mb:
            return

        # Simple LRU: clear cache if over limit
        while self._current_size_mb + size_mb > self._max_size_mb and self._cache:
            oldest_key = next(iter(self._cache))
            oldest_size = self._cache[oldest_key]['size_mb']
            del self._cache[oldest_key]
            self._current_size_mb -= oldest_size

        self._cache[cache_key] = {
            'histogram': histogram,
            'size_mb': size_mb
        }
        self._current_size_mb += size_mb

    def clear(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        self._current_size_mb = 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        return {
            'entries': len(self._cache),
            'size_mb': self._current_size_mb,
            'max_size_mb': self._max_size_mb,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate
        }


class HistogramManager:
    """Histogram computation and management with caching."""

    def __init__(self, cache_size_mb: float = 200.0):
        self._cache = HistogramCache(cache_size_mb)
        self._stats = {
            'total_computations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'computation_time_ms': []
        }

    def compute_histogram_1d(
        self,
        data: np.ndarray,
        bins: Union[int, np.ndarray],
        weights: Optional[np.ndarray] = None,
        density: bool = False,
        params: Optional[HistogramParams] = None
    ) -> Histogram1D:
        """Compute 1D histogram with caching."""
        import time
        start_time = time.perf_counter()

        if params is not None:
            cached = self._cache.get(params.to_dict(), '1d')
            if cached is not None:
                self._stats['cache_hits'] += 1
                return cached

        self._stats['cache_misses'] += 1

        try:
            if len(data) == 0:
                logging.warning("Empty data provided for 1D histogram")
                return Histogram1D(
                    edges=np.array([0.0, 1.0]),
                    counts=np.array([0.0])
                )

            if isinstance(bins, (int, np.integer)):
                data_min, data_max = np.min(data), np.max(data)
                if data_min == data_max:
                    data_min -= 0.5
                    data_max += 0.5
                    logging.debug(f"Zero range in 1D data, expanding to [{data_min}, {data_max}]")
                edges = np.linspace(data_min, data_max, int(bins) + 1)
            else:
                edges = np.asarray(bins, dtype=np.float64)

            counts, edges_out = np.histogram(data, bins=edges, weights=weights, density=density)

            histogram = Histogram1D(
                edges=edges_out,
                counts=counts.astype(np.float64)
            )

            if not histogram.validate():
                raise ValueError("Computed histogram failed validation")

            if params is not None:
                self._cache.put(params.to_dict(), histogram)

            computation_time = (time.perf_counter() - start_time) * 1000
            self._stats['total_computations'] += 1
            self._stats['computation_time_ms'].append(computation_time)

            return histogram

        except Exception as e:
            logging.error(f"Failed to compute 1D histogram: {e}")
            return Histogram1D(
                edges=np.array([0.0, 1.0]),
                counts=np.array([0.0])
            )

    def compute_histogram_2d(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        x_bins: Union[int, np.ndarray],
        y_bins: Union[int, np.ndarray],
        weights: Optional[np.ndarray] = None,
        density: bool = False,
        params: Optional[HistogramParams] = None
    ) -> Histogram2D:
        """Compute 2D histogram with caching."""
        import time
        start_time = time.perf_counter()

        if params is not None:
            cached = self._cache.get(params.to_dict(), '2d')
            if cached is not None:
                self._stats['cache_hits'] += 1
                return cached

        self._stats['cache_misses'] += 1

        try:
            if len(x_data) == 0 or len(y_data) == 0:
                logging.warning("Empty data provided for 2D histogram")
                return Histogram2D(
                    H=np.zeros((1, 1)),
                    x_edges=np.array([0.0, 1.0]),
                    y_edges=np.array([0.0, 1.0])
                )

            if isinstance(x_bins, (int, np.integer)):
                x_min, x_max = np.min(x_data), np.max(x_data)
                if x_min == x_max:
                    x_min -= 0.5
                    x_max += 0.5
                    logging.debug(f"Zero range in X data, expanding to [{x_min}, {x_max}]")
                x_edges = np.linspace(x_min, x_max, int(x_bins) + 1)
            else:
                x_edges = np.asarray(x_bins, dtype=np.float64)

            if isinstance(y_bins, (int, np.integer)):
                y_min, y_max = np.min(y_data), np.max(y_data)
                if y_min == y_max:
                    y_min -= 0.5
                    y_max += 0.5
                    logging.debug(f"Zero range in Y data, expanding to [{y_min}, {y_max}]")
                y_edges = np.linspace(y_min, y_max, int(y_bins) + 1)
            else:
                y_edges = np.asarray(y_bins, dtype=np.float64)

            H, x_edges_out, y_edges_out = np.histogram2d(
                x_data, y_data,
                bins=[x_edges, y_edges],
                weights=weights,
                density=density
            )

            histogram = Histogram2D(
                H=H.astype(np.float64),
                x_edges=x_edges_out,
                y_edges=y_edges_out
            )

            if not histogram.validate():
                raise ValueError("Computed histogram failed validation")

            if params is not None:
                self._cache.put(params.to_dict(), histogram)

            computation_time = (time.perf_counter() - start_time) * 1000
            self._stats['total_computations'] += 1
            self._stats['computation_time_ms'].append(computation_time)

            return histogram

        except Exception as e:
            logging.error(f"Failed to compute 2D histogram: {e}")
            return Histogram2D(
                H=np.zeros((1, 1)),
                x_edges=np.array([0.0, 1.0]),
                y_edges=np.array([0.0, 1.0])
            )

    def clear_cache(self) -> None:
        """Clear histogram cache."""
        self._cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self._cache.get_stats()

    def get_stats(self) -> Dict[str, Any]:
        """Get computation statistics."""
        avg_time = np.mean(self._stats['computation_time_ms']) if self._stats['computation_time_ms'] else 0.0
        return {
            'total_computations': self._stats['total_computations'],
            'cache_hits': self._stats['cache_hits'],
            'cache_misses': self._stats['cache_misses'],
            'avg_computation_time_ms': avg_time,
            'cache': self.get_cache_stats()
        }


# Global histogram manager instance
_histogram_manager: Optional[HistogramManager] = None


def get_histogram_manager() -> HistogramManager:
    """Get or create global histogram manager."""
    global _histogram_manager
    if _histogram_manager is None:
        _histogram_manager = HistogramManager()
    return _histogram_manager


def reset_histogram_manager() -> None:
    """Reset global histogram manager."""
    global _histogram_manager
    _histogram_manager = None
