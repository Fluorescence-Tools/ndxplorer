#!/usr/bin/env python3
"""
Performance optimization utilities for NDxplorer when handling large datasets.
Provides lazy loading, memory-efficient operations, and optimized caching.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Any, List
import threading
import weakref
from functools import lru_cache
import time

import numpy as np
import pandas as pd

from ..logging_config import logging
from ..core.data_source import DataSource

try:  # Optional acceleration via numba
    import numba as nb  # type: ignore

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover - runtime availability only
    nb = None  # type: ignore
    _HAVE_NUMBA = False


class LazyDataLoader:
    """
    Lazy loading wrapper for large datasets.
    
    Provides on-demand data access with memory efficiency and caching.
    
    Parameters
    ----------
    data_source : DataSource
        The underlying data source to wrap
    chunk_size : int, optional
        Size of chunks for lazy loading (default: 10000)
    """
    
    def __init__(self, data_source: DataSource, chunk_size: int = 10000):
        self._data_source = data_source
        self._chunk_size = chunk_size
        self._cache = {}
        self._cache_lock = threading.Lock()
        self._data_hash = None
        
    def _compute_data_hash(self) -> int:
        """Compute a hash of the data for cache invalidation"""
        if self._data_hash is None:
            data = self._data_source.values
            self._data_hash = hash((
                data.shape,
                data.dtype,
                hash(data.tobytes()) if data.size < 1000000 else hash(data[:1000].tobytes())
            ))
        return self._data_hash
        
    def get_chunk(self, start_idx: int, end_idx: int) -> np.ndarray:
        """Get a chunk of data efficiently with caching"""
        chunk_key = (start_idx, end_idx, self._compute_data_hash())
        
        with self._cache_lock:
            if chunk_key in self._cache:
                return self._cache[chunk_key]
                
        # Load chunk from data source
        all_data = self._data_source.values
        chunk = all_data[:, start_idx:end_idx]
        
        with self._cache_lock:
            # Limit cache size to prevent memory issues
            if len(self._cache) > 10:
                # Remove oldest entries
                old_keys = list(self._cache.keys())[:5]
                for key in old_keys:
                    del self._cache[key]
            self._cache[chunk_key] = chunk
            
        return chunk
        
    def invalidate_cache(self):
        """Invalidate all cached chunks"""
        with self._cache_lock:
            self._cache.clear()
            self._data_hash = None


class OptimizedHistogram:
    """
    Memory-efficient histogram computation for large datasets.
    Uses streaming computation and optimized binning strategies.
    """
    
    def __init__(self, data_source: DataSource, max_memory_mb: int = 512):
        self.data_source = data_source
        self.max_memory_mb = max_memory_mb
        self._cached_hist = {}
        
    def compute_histogram_2d_streaming(
        self, 
        x_idx: int, 
        y_idx: int, 
        bins: Tuple[int, int],
        weights: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute 2D histogram using streaming approach for large datasets.
        
        Parameters
        ----------
        x_idx, y_idx : int
            Parameter indices for the histogram
        bins : tuple of int
            Number of bins for x and y dimensions
        weights : np.ndarray, optional
            Weights for histogram computation
            
        Returns
        -------
        H : np.ndarray
            2D histogram counts
        x_edges, y_edges : np.ndarray
            Bin edges for x and y dimensions
        """
        data = self.data_source.values
        n_points = data.shape[1]
        
        # Estimate memory usage and choose strategy
        estimated_memory = n_points * 8 * 3  # 3 arrays of float64
        if estimated_memory > self.max_memory_mb * 1024 * 1024:
            return self._compute_histogram_chunked(x_idx, y_idx, bins, weights)
        else:
            return self._compute_histogram_direct(x_idx, y_idx, bins, weights)
            
    def _compute_histogram_direct(
        self, 
        x_idx: int, 
        y_idx: int, 
        bins: Tuple[int, int],
        weights: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Direct computation for smaller datasets."""
        data = self.data_source.values
        x_data = data[x_idx, :]
        y_data = data[y_idx, :]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            H, x_edges, y_edges = np.histogram2d(
                x=x_data, y=y_data, bins=bins, weights=weights, density=False
            )
        H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
        return H, x_edges, y_edges
        
    def _compute_histogram_chunked(
        self, 
        x_idx: int, 
        y_idx: int, 
        bins: Tuple[int, int],
        weights: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Chunked computation for large datasets."""
        data = self.data_source.values
        n_points = data.shape[1]
        chunk_size = min(50000, n_points // 10)  # Adaptive chunk size
        
        # Initialize accumulator
        H_accum = np.zeros(bins, dtype=np.float64)
        x_min, x_max = np.min(data[x_idx, :]), np.max(data[x_idx, :])
        y_min, y_max = np.min(data[y_idx, :]), np.max(data[y_idx, :])
        
        # Process in chunks
        for start in range(0, n_points, chunk_size):
            end = min(start + chunk_size, n_points)
            x_chunk = data[x_idx, start:end]
            y_chunk = data[y_idx, start:end]
            w_chunk = weights[start:end] if weights is not None else None
            
            with np.errstate(divide='ignore', invalid='ignore'):
                H_chunk, _, _ = np.histogram2d(
                    x=x_chunk, y=y_chunk, bins=bins, weights=w_chunk, density=False
                )
            H_accum += np.nan_to_num(H_chunk, nan=0.0, posinf=0.0, neginf=0.0)
            
        # Generate bin edges
        x_edges = np.linspace(x_min, x_max, bins[0] + 1)
        y_edges = np.linspace(y_min, y_max, bins[1] + 1)
        
        return H_accum, x_edges, y_edges


class PerformanceMonitor:
    """Monitor and log performance metrics for NDxplorer operations."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
        
    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.start_times[operation] = time.time()
        
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration."""
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            self.metrics[operation] = duration
            del self.start_times[operation]
            logging.info(f"Operation '{operation}' completed in {duration:.3f}s")
            return duration
        return 0.0
        
    def log_memory_usage(self, operation: str):
        """Log current memory usage for an operation."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            logging.info(f"Memory usage for '{operation}': {memory_mb:.1f} MB")
        except ImportError:
            pass
            
    def get_performance_summary(self) -> dict:
        """Get summary of all recorded metrics."""
        return self.metrics.copy()


_PERF_MONITOR: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Return a shared PerformanceMonitor instance."""
    global _PERF_MONITOR
    if _PERF_MONITOR is None:
        _PERF_MONITOR = PerformanceMonitor()
    return _PERF_MONITOR


# Utility functions for common performance optimizations

def optimize_dtype(data: np.ndarray) -> np.ndarray:
    """
    Optimize data type to reduce memory usage while preserving precision.
    
    Parameters
    ----------
    data : np.ndarray
        Input data to optimize
        
    Returns
    -------
    np.ndarray
        Data with optimized dtype
    """
    if data.dtype == np.float64:
        # Check if we can use float32 without losing precision
        data_float32 = data.astype(np.float32)
        if np.allclose(data, data_float32, rtol=1e-6):
            return data_float32
    elif data.dtype == np.int64:
        # Check if we can use int32
        if np.all(data >= np.iinfo(np.int32).min) and np.all(data <= np.iinfo(np.int32).max):
            return data.astype(np.int32)
    return data


def downsample_data(data: np.ndarray, target_size: int = 1000000) -> np.ndarray:
    """
    Downsample data to target size for faster visualization.
    
    Parameters
    ----------
    data : np.ndarray
        Input data with shape (n_parameters, n_points)
    target_size : int
        Target number of points
        
    Returns
    -------
    np.ndarray
        Downsampled data
    """
    n_params, n_points = data.shape
    if n_points <= target_size:
        return data
        
    # Random downsampling
    indices = np.random.choice(n_points, target_size, replace=False)
    return data[:, indices]


def create_memory_efficient_mask(data_shape: Tuple[int, int]) -> np.ndarray:
    """
    Create a memory-efficient boolean mask.
    
    Parameters
    ----------
    data_shape : tuple
        Shape of the data (n_parameters, n_points)
        
    Returns
    -------
    np.ndarray
        Boolean mask with optimized dtype
    """
    # Use np.bool8 instead of np.bool for memory efficiency
    return np.zeros(data_shape, dtype=np.bool8)


# ---------------------------
# Adaptive histogram helpers
# ---------------------------

DEFAULT_STREAM_THRESHOLD = int(os.getenv("NDX_HISTOGRAM_STREAM_THRESHOLD", "750000"))
DEFAULT_CHUNK_SIZE = int(os.getenv("NDX_HISTOGRAM_CHUNK_SIZE", "100000"))

# Target dtype for histogram data - float32 saves 50% memory vs float64
# with sufficient precision for binning operations
HISTOGRAM_DTYPE = np.float32

if _HAVE_NUMBA:

    @nb.njit(cache=True, parallel=True, fastmath=True)  # type: ignore[misc]
    def _histogram1d_numba(
        data: np.ndarray,
        edges: np.ndarray,
        weights: np.ndarray,
        has_weights: bool,
    ) -> np.ndarray:
        """Numba-accelerated 1D histogram computation (float32 optimized)."""
        n_bins = edges.size - 1
        # Use float32 accumulator - sufficient precision for counts
        H = np.zeros(n_bins, dtype=np.float32)
        n = data.size
        
        for i in nb.prange(n):
            v = data[i]
            if np.isnan(v):
                continue
            # Binary search for bin
            bi = np.searchsorted(edges, v, side='right') - 1
            if bi < 0 or bi >= n_bins:
                continue
            w = np.float32(weights[i]) if has_weights else np.float32(1.0)
            H[bi] += w
        
        return H

    @nb.njit(cache=True, fastmath=False)  # type: ignore[misc]
    def _histogram2d_numba(
        x: np.ndarray,
        y: np.ndarray,
        x_edges: np.ndarray,
        y_edges: np.ndarray,
        weights: np.ndarray,
        has_weights: bool,
    ) -> np.ndarray:
        """Numba-accelerated 2D histogram (float32 optimized)."""
        nx = x_edges.size - 1
        ny = y_edges.size - 1
        # Use float32 accumulator - sufficient for count data
        H = np.zeros((nx, ny), dtype=np.float32)

        for i in range(x.size):
            xv = x[i]
            yv = y[i]

            if np.isnan(xv) or np.isnan(yv):
                continue

            xi = np.searchsorted(x_edges, xv, side="right") - 1
            yi = np.searchsorted(y_edges, yv, side="right") - 1

            if xi < 0 or xi >= nx or yi < 0 or yi >= ny:
                continue

            w = np.float32(weights[i]) if has_weights else np.float32(1.0)
            H[xi, yi] += w

        return H

else:  # pragma: no cover - exercised only without numba installed
    _histogram1d_numba = None  # type: ignore
    _histogram2d_numba = None  # type: ignore


@dataclass
class Histogram2DComputation:
    """Container for adaptive histogram results."""

    data: Tuple[np.ndarray, np.ndarray, np.ndarray]
    chunked: bool
    n_points: int


def _resolve_threshold(value: Optional[int], fallback: int) -> int:
    try:
        return int(value) if value is not None else fallback
    except (TypeError, ValueError):
        return fallback


def compute_histogram2d_adaptive(
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
    threshold: Optional[int] = None,
    chunk_size: Optional[int] = None,
) -> Histogram2DComputation:
    """
    Compute a 2D histogram using chunked accumulation when datasets exceed a threshold.

    Parameters
    ----------
    x_data, y_data : np.ndarray
        1D arrays containing the axis data (already masked/filtered).
    x_edges, y_edges : np.ndarray
        Pre-sanitized bin edges for the histogram.
    weights : np.ndarray, optional
        Optional weights aligned with the data arrays.
    threshold : int, optional
        Override the streaming threshold (defaults to env `NDX_HISTOGRAM_STREAM_THRESHOLD` or 750k points).
    chunk_size : int, optional
        Override the chunk size used when streaming (env `NDX_HISTOGRAM_CHUNK_SIZE`, default 100k).

    Returns
    -------
    Histogram2DComputation
        Contains the histogram tuple plus metadata describing whether chunking was used.
    """
    n_points = len(x_data)
    threshold = _resolve_threshold(threshold, DEFAULT_STREAM_THRESHOLD)
    chunk_size = max(1000, _resolve_threshold(chunk_size, DEFAULT_CHUNK_SIZE))

    # Use float32 for data to halve memory usage
    x_edges = np.asarray(x_edges, dtype=np.float32)
    y_edges = np.asarray(y_edges, dtype=np.float32)
    x_data = np.asarray(x_data, dtype=np.float32)
    y_data = np.asarray(y_data, dtype=np.float32)

    if n_points <= threshold:
        with np.errstate(divide="ignore", invalid="ignore"):
            # numpy histogram2d returns float64, but we convert back to float32
            H, x_out, y_out = np.histogram2d(
                x_data,
                y_data,
                bins=[x_edges, y_edges],
                weights=weights.astype(np.float32) if weights is not None else None,
                density=False,
            )
        H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return Histogram2DComputation((H, x_out.astype(np.float32), y_out.astype(np.float32)), chunked=False, n_points=n_points)

    if _HAVE_NUMBA and _histogram2d_numba is not None:
        logging.info(
            "compute_histogram2d_adaptive: numba streaming %d points", n_points
        )
        # Use float32 throughout for memory efficiency
        nb_weights = (
            np.asarray(weights, dtype=np.float32) if weights is not None else np.empty(1, dtype=np.float32)
        )
        H_accum = _histogram2d_numba(
            x_data,
            y_data,
            x_edges,
            y_edges,
            nb_weights,
            weights is not None,
        )
        return Histogram2DComputation((H_accum, x_edges, y_edges), chunked=True, n_points=n_points)

    logging.info(
        "compute_histogram2d_adaptive: streaming %d points with chunk_size=%d",
        n_points,
        chunk_size,
    )
    # Preallocate accumulator with float32 for memory efficiency
    H_accum = np.zeros((len(x_edges) - 1, len(y_edges) - 1), dtype=np.float32)

    for start in range(0, n_points, chunk_size):
        end = min(start + chunk_size, n_points)
        w_chunk = weights[start:end].astype(np.float32) if weights is not None else None
        with np.errstate(divide="ignore", invalid="ignore"):
            H_chunk, _, _ = np.histogram2d(
                x_data[start:end],
                y_data[start:end],
                bins=[x_edges, y_edges],
                weights=w_chunk,
                density=False,
            )
        H_accum += np.nan_to_num(H_chunk, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return Histogram2DComputation((H_accum, x_edges, y_edges), chunked=True, n_points=n_points)


@dataclass
class Histogram1DComputation:
    """Container for adaptive 1D histogram results."""
    counts: np.ndarray
    edges: np.ndarray
    n_points: int
    used_numba: bool


def compute_histogram1d_adaptive(
    data: np.ndarray,
    edges: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
    density: bool = False,
) -> Histogram1DComputation:
    """
    Compute 1D histogram using Numba when available for large datasets.

    Parameters
    ----------
    data : np.ndarray
        1D array of values.
    edges : np.ndarray
        Pre-computed bin edges.
    weights : np.ndarray, optional
        Optional weights.
    density : bool
        If True, normalize to density.

    Returns
    -------
    Histogram1DComputation
        Contains counts, edges, and metadata.
    """
    n_points = len(data)
    # Use float32 for memory efficiency
    edges = np.asarray(edges, dtype=np.float32)
    data = np.asarray(data, dtype=np.float32)

    used_numba = False
    if _HAVE_NUMBA and _histogram1d_numba is not None and n_points > 50000:
        # Use float32 weights for consistency
        nb_weights = (
            np.asarray(weights, dtype=np.float32) if weights is not None else np.empty(1, dtype=np.float32)
        )
        counts = _histogram1d_numba(data, edges, nb_weights, weights is not None)
        used_numba = True
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            counts, _ = np.histogram(data, bins=edges, weights=weights, density=False)
        counts = np.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if density and counts.sum() > 0:
        bin_widths = np.diff(edges)
        counts = counts / (counts.sum() * bin_widths)

    return Histogram1DComputation(counts=counts, edges=edges, n_points=n_points, used_numba=used_numba)


# ---------------------------
# Display downsampling for large datasets
# ---------------------------

DEFAULT_DISPLAY_SAMPLE_SIZE = int(os.getenv("NDX_DISPLAY_SAMPLE_SIZE", "500000"))


def downsample_for_display(
    data: np.ndarray,
    max_points: Optional[int] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, bool]:
    """
    Downsample data for faster interactive display.

    For very large datasets, use a representative random sample for
    interactive histogram updates, then compute full data on final.

    Parameters
    ----------
    data : np.ndarray
        Input data array (can be 1D or 2D with shape (n_params, n_points)).
    max_points : int, optional
        Maximum points to keep (default from env NDX_DISPLAY_SAMPLE_SIZE or 500k).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    sampled_data : np.ndarray
        Downsampled data (same shape structure).
    was_downsampled : bool
        True if downsampling was applied.
    """
    max_points = max_points or DEFAULT_DISPLAY_SAMPLE_SIZE
    
    if data.ndim == 1:
        n_points = len(data)
        if n_points <= max_points:
            return data, False
        rng = np.random.default_rng(seed)
        indices = rng.choice(n_points, max_points, replace=False)
        return data[indices], True
    elif data.ndim == 2:
        n_points = data.shape[1]
        if n_points <= max_points:
            return data, False
        rng = np.random.default_rng(seed)
        indices = rng.choice(n_points, max_points, replace=False)
        return data[:, indices], True
    else:
        return data, False


# ---------------------------
# Parallel histogram helpers
# ---------------------------

def compute_histograms_parallel(
    x_data: np.ndarray,
    y_data: np.ndarray,
    z_data: np.ndarray,
    x_bins: np.ndarray,
    y_bins: np.ndarray,
    z_bins: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
    normed_x: bool = False,
    normed_y: bool = False,
    normed_z: bool = False,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Compute X, Y, Z histograms in parallel using threading.

    Returns
    -------
    tuple of (edges, counts) for x, y, z
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def compute_one(data, bins, normed):
        result = compute_histogram1d_adaptive(data, bins, weights=weights, density=normed)
        return result.edges, result.counts

    results = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(compute_one, x_data, x_bins, normed_x): 'x',
            executor.submit(compute_one, y_data, y_bins, normed_y): 'y',
            executor.submit(compute_one, z_data, z_bins, normed_z): 'z',
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as e:
                logging.warning("Parallel histogram error for %s: %s", key, e)
                # Fallback
                if key == 'x':
                    results[key] = (x_bins, np.zeros(len(x_bins) - 1))
                elif key == 'y':
                    results[key] = (y_bins, np.zeros(len(y_bins) - 1))
                else:
                    results[key] = (z_bins, np.zeros(len(z_bins) - 1))

    return results.get('x'), results.get('y'), results.get('z')


def _generate_synthetic_data(n_points: int, seed: int = 13) -> Tuple[np.ndarray, np.ndarray]:
    """Return correlated synthetic x/y arrays for benchmarking."""
    rng = np.random.default_rng(seed)
    x = rng.normal(loc=0.0, scale=1.0, size=n_points).astype(np.float64)
    noise = rng.normal(loc=0.0, scale=0.75, size=n_points).astype(np.float64)
    y = 0.45 * x + noise
    return x, y


def run_histogram_benchmark(
    *,
    n_points: int = 1_000_000,
    bins: int = 200,
    repeats: int = 3,
    seed: int = 42,
    threshold: Optional[int] = None,
    chunk_size: Optional[int] = None,
) -> dict:
    """
    Execute an end-to-end histogram benchmark using synthetic data.

    Parameters
    ----------
    n_points : int
        Number of synthetic samples per axis (default: 1e6).
    bins : int
        Number of bins per axis (uniform linspace).
    repeats : int
        Number of benchmark iterations to average.
    seed : int
        RNG seed for reproducible data.
    threshold : Optional[int]
        Override adaptive chunking threshold (defaults to env or 750k).
    chunk_size : Optional[int]
        Override adaptive chunk size (defaults to env or 100k).

    Returns
    -------
    dict
        Summary containing durations, chunking flag, and benchmark parameters.
    """
    n_points = int(n_points)
    bins = max(10, int(bins))
    repeats = max(1, int(repeats))
    threshold = _resolve_threshold(threshold, DEFAULT_STREAM_THRESHOLD)
    chunk_size = max(1000, _resolve_threshold(chunk_size, DEFAULT_CHUNK_SIZE))

    x_data, y_data = _generate_synthetic_data(n_points, seed=seed)
    x_edges = np.linspace(np.min(x_data), np.max(x_data), bins + 1, dtype=np.float64)
    y_edges = np.linspace(np.min(y_data), np.max(y_data), bins + 1, dtype=np.float64)

    monitor = get_performance_monitor()
    durations = []
    hist_meta: Optional[Histogram2DComputation] = None

    for idx in range(repeats):
        label = f"ndx_histogram_benchmark_{idx + 1}"
        monitor.start_timer(label)
        hist_meta = compute_histogram2d_adaptive(
            x_data,
            y_data,
            x_edges,
            y_edges,
            threshold=threshold,
            chunk_size=chunk_size,
        )
        monitor.log_memory_usage(label)
        duration = monitor.end_timer(label)
        durations.append(duration)

    avg_duration = float(np.mean(durations)) if durations else 0.0
    summary = {
        "n_points": n_points,
        "bins_per_axis": bins,
        "repeats": repeats,
        "durations_s": durations,
        "avg_duration_s": avg_duration,
        "threshold": threshold,
        "chunk_size": chunk_size,
        "chunked": bool(hist_meta.chunked) if hist_meta else False,
        "total_points_last_run": hist_meta.n_points if hist_meta else n_points,
    }

    logging.info(
        "Histogram benchmark: n_points=%d bins=%d repeats=%d avg=%.3fs chunked=%s threshold=%d chunk_size=%d",
        n_points,
        bins,
        repeats,
        avg_duration,
        summary["chunked"],
        threshold,
        chunk_size,
    )
    return summary


def _build_benchmark_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NDXplorer histogram performance benchmark (synthetic data)."
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=1_000_000,
        help="Number of synthetic points per axis (default: 1_000_000).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=200,
        help="Number of bins per axis (default: 200).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of benchmark iterations to average (default: 3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducible synthetic data (default: 42).",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="Override adaptive chunk threshold (defaults to env/750k).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override chunk size for streaming path (defaults to env/100k).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON summary instead of formatted text.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> dict:
    """Entry point for `python -m ndxplorer.utils.performance_optimizations`."""
    parser = _build_benchmark_parser()
    args = parser.parse_args(argv)
    summary = run_histogram_benchmark(
        n_points=args.n_points,
        bins=args.bins,
        repeats=args.repeats,
        seed=args.seed,
        threshold=args.threshold,
        chunk_size=args.chunk_size,
    )
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(
            (
                "Histogram benchmark\n"
                f"  points       : {summary['n_points']}\n"
                f"  bins/axis    : {summary['bins_per_axis']}\n"
                f"  repeats      : {summary['repeats']}\n"
                f"  durations (s): {', '.join(f'{d:.3f}' for d in summary['durations_s'])}\n"
                f"  avg (s)      : {summary['avg_duration_s']:.3f}\n"
                f"  chunked      : {summary['chunked']}\n"
                f"  threshold    : {summary['threshold']}\n"
                f"  chunk size   : {summary['chunk_size']}"
            )
        )
    return summary


if __name__ == "__main__":
    main()
