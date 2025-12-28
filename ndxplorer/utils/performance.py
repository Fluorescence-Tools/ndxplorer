"""
Main performance optimization module for ndxplorer.

This module provides a unified interface to all performance optimizations:
- Bitfield masks (8x memory reduction)
- Histogram caching (2-5x speedup on repeated operations)
- Numba acceleration (2-10x speedup on computations)
- Vectorized operations (SIMD-optimized)

Usage Examples
--------------

Basic usage (automatic optimization):
    >>> from ndxplorer.utils.performance import optimize_ndxplorer
    >>> optimize_ndxplorer(ndxplorer_instance)

High-performance mode:
    >>> from ndxplorer.utils.performance import enable_high_performance
    >>> enable_high_performance()

Low-memory mode (for large datasets):
    >>> from ndxplorer.utils.performance import enable_low_memory
    >>> enable_low_memory()

Manual cache control:
    >>> from ndxplorer.utils.performance import get_cache_stats, clear_caches
    >>> stats = get_cache_stats()
    >>> print(stats)
    >>> clear_caches()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional
import numpy as np

from ..logging_config import logging

# Import performance modules
try:
    import boost_histogram as bh
    _HAVE_BOOST_HISTOGRAM = True
except ImportError:
    bh = None
    _HAVE_BOOST_HISTOGRAM = False

try:
    from .bitfield_mask import BitfieldMask, create_mask_from_condition
    _HAVE_BITFIELD = True
except ImportError:
    _HAVE_BITFIELD = False
    BitfieldMask = None
    create_mask_from_condition = None

try:
    from .cache_manager import get_cache_manager, clear_all_caches
    _HAVE_CACHE = True
except ImportError:
    _HAVE_CACHE = False
    get_cache_manager = None
    clear_all_caches = None

try:
    from .fast_histogram import fast_histogram_1d, fast_histogram_2d
    _HAVE_FAST_HISTOGRAM = True
except ImportError:
    _HAVE_FAST_HISTOGRAM = False
    fast_histogram_1d = None
    fast_histogram_2d = None

try:
    from .vectorized_ops import (
        fast_percentile_range,
        fast_minmax,
        fast_rectangular_selection,
        combine_masks_fast
    )
    _HAVE_VECTORIZED = True
except ImportError:
    _HAVE_VECTORIZED = False
    fast_percentile_range = None
    fast_minmax = None
    fast_rectangular_selection = None
    combine_masks_fast = None

try:
    from .performance_config import (
        get_performance_config,
        set_performance_config,
        enable_high_performance,
        enable_low_memory,
        enable_balanced,
        PerformanceConfig
    )
    _HAVE_CONFIG = True
except ImportError:
    _HAVE_CONFIG = False
    get_performance_config = None
    set_performance_config = None
    enable_high_performance = None
    enable_low_memory = None
    enable_balanced = None
    PerformanceConfig = None

if TYPE_CHECKING:
    from ..core.plot_main import NDXplorer


# ---- High-level optimization functions ----

def optimize_ndxplorer(ndxplorer: "NDXplorer", mode: str = "balanced") -> None:
    """
    Apply performance optimizations to an NDXplorer instance.
    
    Parameters
    ----------
    ndxplorer : NDXplorer
        NDXplorer instance to optimize
    mode : str
        Optimization mode: 'balanced', 'high_performance', or 'low_memory'
    """
    if mode == "high_performance" and _HAVE_CONFIG and enable_high_performance:
        enable_high_performance()
    elif mode == "low_memory" and _HAVE_CONFIG and enable_low_memory:
        enable_low_memory()
    elif mode == "balanced" and _HAVE_CONFIG and enable_balanced:
        enable_balanced()
    
    # Mark instance as optimized
    ndxplorer._performance_optimized = True
    
    logging.info(f"[Performance] NDXplorer optimized with mode: {mode}")
    log_available_features()


def log_available_features() -> None:
    """Log which performance features are available."""
    features = {
        "Boost-histogram (fastest)": _HAVE_BOOST_HISTOGRAM,
        "Bitfield masks": _HAVE_BITFIELD,
        "Histogram caching": _HAVE_CACHE,
        "Fast histogram": _HAVE_FAST_HISTOGRAM,
        "Vectorized ops": _HAVE_VECTORIZED,
        "Performance config": _HAVE_CONFIG,
    }
    
    logging.info("[Performance] Available features:")
    for feature, available in features.items():
        status = "✓" if available else "✗"
        logging.info(f"  {status} {feature}")


def get_cache_stats() -> dict:
    """Get statistics for all caches."""
    if _HAVE_CACHE and get_cache_manager:
        manager = get_cache_manager()
        return manager.stats()
    return {}


def clear_caches() -> None:
    """Clear all performance caches."""
    if _HAVE_CACHE and clear_all_caches:
        clear_all_caches()
        logging.info("[Performance] All caches cleared")
    else:
        logging.warning("[Performance] Cache system not available")


def log_cache_stats() -> None:
    """Log cache statistics."""
    stats = get_cache_stats()
    if stats:
        logging.info("[Performance] Cache statistics:")
        for cache_name, cache_stats in stats.items():
            logging.info(f"  {cache_name}:")
            for key, value in cache_stats.items():
                logging.info(f"    {key}: {value}")
    else:
        logging.info("[Performance] No cache statistics available")


# ---- Optimized histogram wrappers ----

def compute_histogram_1d_optimized(
    data: np.ndarray,
    bins: np.ndarray | int,
    weights: Optional[np.ndarray] = None,
    density: bool = False
) -> tuple:
    """
    Compute 1D histogram with all available optimizations.
    
    Automatically uses caching and Numba acceleration when available.
    """
    if _HAVE_FAST_HISTOGRAM and fast_histogram_1d:
        return fast_histogram_1d(data, bins, weights, density, use_cache=True, use_numba=True)
    else:
        # Fallback to numpy
        with np.errstate(divide='ignore', invalid='ignore'):
            counts, edges = np.histogram(data, bins=bins, weights=weights, density=density)
        return counts, edges


def compute_histogram_2d_optimized(
    x: np.ndarray,
    y: np.ndarray,
    bins: list | tuple,
    weights: Optional[np.ndarray] = None,
    density: bool = False
) -> tuple:
    """
    Compute 2D histogram with all available optimizations.
    
    Automatically uses caching and Numba acceleration when available.
    """
    if _HAVE_FAST_HISTOGRAM and fast_histogram_2d:
        return fast_histogram_2d(x, y, bins, weights, density, use_cache=True, use_numba=True)
    else:
        # Fallback to numpy
        with np.errstate(divide='ignore', invalid='ignore'):
            H, x_edges, y_edges = np.histogram2d(x, y, bins=bins, weights=weights, density=density)
        return H, x_edges, y_edges


# ---- Optimized statistics ----

def compute_percentile_range_optimized(
    data: np.ndarray,
    low_pct: float = 1.0,
    high_pct: float = 99.0,
    mask: Optional[np.ndarray] = None
) -> tuple[float, float]:
    """
    Compute percentile range with optimization.
    
    Uses fast partial-sort algorithm instead of full sort.
    """
    if _HAVE_VECTORIZED and fast_percentile_range:
        return fast_percentile_range(data, low_pct, high_pct, mask)
    else:
        # Fallback
        if mask is not None:
            valid = data[~mask]
        else:
            valid = data
        valid = valid[np.isfinite(valid)]
        if len(valid) == 0:
            return 0.0, 1.0
        vmin = float(np.percentile(valid, low_pct))
        vmax = float(np.percentile(valid, high_pct))
        return vmin, vmax


# ---- Memory usage reporting ----

def estimate_memory_usage(ndxplorer: "NDXplorer") -> dict:
    """
    Estimate memory usage of ndxplorer components.
    
    Returns
    -------
    usage : dict
        Memory usage breakdown in MB
    """
    usage = {}
    
    # Data source
    if hasattr(ndxplorer, '_data_source') and ndxplorer._data_source is not None:
        if hasattr(ndxplorer._data_source, 'values'):
            try:
                values = ndxplorer._data_source.values
                usage['data_values_mb'] = values.nbytes / (1024 * 1024)
            except Exception:
                pass
    
    # Cached values
    if hasattr(ndxplorer, '_cached_values') and ndxplorer._cached_values is not None:
        if isinstance(ndxplorer._cached_values, np.ndarray):
            usage['cached_values_mb'] = ndxplorer._cached_values.nbytes / (1024 * 1024)
        elif _HAVE_BITFIELD and isinstance(ndxplorer._cached_values, BitfieldMask):
            usage['cached_values_mb'] = ndxplorer._cached_values.nbytes / (1024 * 1024)
    
    # Cache manager
    cache_stats = get_cache_stats()
    if cache_stats:
        for cache_name, stats in cache_stats.items():
            if 'memory_mb' in stats:
                usage[f'{cache_name}_cache_mb'] = stats['memory_mb']
    
    # Total
    usage['total_mb'] = sum(v for k, v in usage.items() if k.endswith('_mb'))
    
    return usage


def log_memory_usage(ndxplorer: "NDXplorer") -> None:
    """Log memory usage breakdown."""
    usage = estimate_memory_usage(ndxplorer)
    logging.info("[Performance] Memory usage:")
    for key, value in usage.items():
        logging.info(f"  {key}: {value:.2f} MB")


# ---- Performance benchmarking ----

def benchmark_histogram(
    data_size: int = 1000000,
    n_bins: int = 100,
    n_iterations: int = 10
) -> dict:
    """
    Benchmark histogram computation performance.
    
    Returns
    -------
    results : dict
        Timing results for different methods
    """
    import time
    
    # Generate test data
    data = np.random.randn(data_size)
    
    results = {}
    
    # Numpy baseline
    times = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        np.histogram(data, bins=n_bins)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    results['numpy_mean_ms'] = np.mean(times) * 1000
    results['numpy_std_ms'] = np.std(times) * 1000
    
    # Optimized version
    if _HAVE_FAST_HISTOGRAM and fast_histogram_1d:
        times = []
        for _ in range(n_iterations):
            t0 = time.perf_counter()
            fast_histogram_1d(data, n_bins, use_cache=False, use_numba=True)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        results['optimized_mean_ms'] = np.mean(times) * 1000
        results['optimized_std_ms'] = np.std(times) * 1000
        results['speedup'] = results['numpy_mean_ms'] / results['optimized_mean_ms']
    
    return results


def log_benchmark_results(data_size: int = 1000000) -> None:
    """Run and log benchmark results."""
    logging.info(f"[Performance] Running benchmark with {data_size} points...")
    results = benchmark_histogram(data_size=data_size)
    logging.info("[Performance] Benchmark results:")
    for key, value in results.items():
        if 'speedup' in key:
            logging.info(f"  {key}: {value:.2f}x")
        else:
            logging.info(f"  {key}: {value:.2f}")


# ---- Exports ----

__all__ = [
    # Main functions
    'optimize_ndxplorer',
    'log_available_features',
    'get_cache_stats',
    'clear_caches',
    'log_cache_stats',
    
    # Optimized computations
    'compute_histogram_1d_optimized',
    'compute_histogram_2d_optimized',
    'compute_percentile_range_optimized',
    
    # Memory and benchmarking
    'estimate_memory_usage',
    'log_memory_usage',
    'benchmark_histogram',
    'log_benchmark_results',
    
    # Configuration
    'enable_high_performance',
    'enable_low_memory',
    'enable_balanced',
]
