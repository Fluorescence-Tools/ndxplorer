"""
Advanced caching system for ndxplorer computations.

Provides multi-level caching with LRU eviction, memory-aware limits,
and hash-based invalidation for expensive operations like histograms.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import Any, Optional, Tuple, Callable
import numpy as np

from ..logging_config import logging


class CacheEntry:
    """Single cache entry with metadata."""
    
    __slots__ = ('value', 'timestamp', 'hits', 'size_bytes')
    
    def __init__(self, value: Any, size_bytes: int = 0):
        self.value = value
        self.timestamp = time.perf_counter()
        self.hits = 0
        self.size_bytes = size_bytes


class LRUCache:
    """
    Least Recently Used cache with memory limits.
    
    Features:
    - Automatic eviction when memory limit exceeded
    - Access count tracking
    - Fast hash-based lookups
    - Memory-aware capacity management
    """
    
    def __init__(self, max_memory_mb: float = 100.0, max_entries: int = 100):
        """
        Parameters
        ----------
        max_memory_mb : float
            Maximum memory usage in MB before eviction
        max_entries : int
            Maximum number of entries before eviction
        """
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self._max_entries = max_entries
        self._current_memory = 0
        self._hits = 0
        self._misses = 0
    
    def _compute_size(self, value: Any) -> int:
        """Estimate memory size of value in bytes."""
        if isinstance(value, np.ndarray):
            return value.nbytes
        elif isinstance(value, tuple):
            return sum(self._compute_size(v) for v in value)
        elif isinstance(value, dict):
            return sum(self._compute_size(k) + self._compute_size(v) for k, v in value.items())
        else:
            # Rough estimate for other types
            return 64
    
    def _evict_if_needed(self, incoming_size: int) -> None:
        """Evict LRU entries until we have space."""
        while (
            self._cache and
            (len(self._cache) >= self._max_entries or 
             self._current_memory + incoming_size > self._max_memory_bytes)
        ):
            key, entry = self._cache.popitem(last=False)
            self._current_memory -= entry.size_bytes
            logging.debug(f"[LRUCache] Evicted key {key[:16]}... (freed {entry.size_bytes} bytes)")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache, updating access order."""
        if key in self._cache:
            self._hits += 1
            entry = self._cache[key]
            entry.hits += 1
            entry.timestamp = time.perf_counter()
            # Move to end (most recent)
            self._cache.move_to_end(key)
            return entry.value
        self._misses += 1
        return None
    
    def put(self, key: str, value: Any) -> None:
        """Add or update cache entry."""
        size = self._compute_size(value)
        
        # Remove old entry if updating
        if key in self._cache:
            old_entry = self._cache.pop(key)
            self._current_memory -= old_entry.size_bytes
        
        self._evict_if_needed(size)
        
        entry = CacheEntry(value, size)
        self._cache[key] = entry
        self._current_memory += size
    
    def invalidate(self, key: str) -> None:
        """Remove specific entry from cache."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._current_memory -= entry.size_bytes
    
    def clear(self) -> None:
        """Clear entire cache."""
        self._cache.clear()
        self._current_memory = 0
        self._hits = 0
        self._misses = 0
    
    def stats(self) -> dict:
        """Return cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            'entries': len(self._cache),
            'memory_mb': self._current_memory / (1024 * 1024),
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
        }


class HistogramCache:
    """
    Specialized cache for histogram computations.
    
    Uses content-aware hashing to detect when recomputation is needed.
    Tracks data shape, bins, weights, and mask state.
    """
    
    def __init__(self, max_memory_mb: float = 200.0):
        self._cache = LRUCache(max_memory_mb=max_memory_mb, max_entries=50)
    
    def _make_key(
        self,
        data: np.ndarray,
        bins: np.ndarray | int,
        weights: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        normed: bool = False,
        histogram_type: str = '1d'
    ) -> str:
        """
        Create cache key from histogram parameters.
        
        Uses fast hashing to avoid expensive comparisons.
        """
        h = hashlib.blake2b(digest_size=16)
        
        # Data shape and dtype
        h.update(str(data.shape).encode())
        h.update(str(data.dtype).encode())
        
        # Sample data hash (first/last/middle values for speed)
        if data.size > 0:
            indices = [0, data.size // 2, data.size - 1] if data.size > 2 else list(range(data.size))
            sample = data.flat[indices]
            h.update(sample.tobytes())
        
        # Bins
        if isinstance(bins, (int, np.integer)):
            h.update(str(bins).encode())
        else:
            h.update(bins.tobytes())
        
        # Weights
        if weights is not None:
            h.update(b'weighted')
            if weights.size > 0:
                indices = [0, weights.size // 2, weights.size - 1] if weights.size > 2 else list(range(weights.size))
                sample = weights.flat[indices]
                h.update(sample.tobytes())
        
        # Mask
        if mask is not None:
            h.update(b'masked')
            h.update(str(np.sum(mask)).encode())
        
        # Other params
        h.update(str(normed).encode())
        h.update(histogram_type.encode())
        
        return h.hexdigest()
    
    def get_histogram_1d(
        self,
        data: np.ndarray,
        bins: np.ndarray | int,
        weights: Optional[np.ndarray] = None,
        normed: bool = False,
        mask: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get cached 1D histogram if available."""
        key = self._make_key(data, bins, weights, mask, normed, '1d')
        return self._cache.get(key)
    
    def put_histogram_1d(
        self,
        data: np.ndarray,
        bins: np.ndarray | int,
        result: Tuple[np.ndarray, np.ndarray],
        weights: Optional[np.ndarray] = None,
        normed: bool = False,
        mask: Optional[np.ndarray] = None,
    ) -> None:
        """Cache 1D histogram result."""
        key = self._make_key(data, bins, weights, mask, normed, '1d')
        self._cache.put(key, result)
    
    def get_histogram_2d(
        self,
        x: np.ndarray,
        y: np.ndarray,
        bins: list | Tuple,
        weights: Optional[np.ndarray] = None,
        normed: bool = False,
        mask: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Get cached 2D histogram if available."""
        # Combine x and y for hashing
        combined = np.column_stack([x, y])
        bins_array = np.array(bins, dtype=object)
        key = self._make_key(combined, bins_array, weights, mask, normed, '2d')
        return self._cache.get(key)
    
    def put_histogram_2d(
        self,
        x: np.ndarray,
        y: np.ndarray,
        bins: list | Tuple,
        result: Tuple[np.ndarray, np.ndarray, np.ndarray],
        weights: Optional[np.ndarray] = None,
        normed: bool = False,
        mask: Optional[np.ndarray] = None,
    ) -> None:
        """Cache 2D histogram result."""
        combined = np.column_stack([x, y])
        bins_array = np.array(bins, dtype=object)
        key = self._make_key(combined, bins_array, weights, mask, normed, '2d')
        self._cache.put(key, result)
    
    def clear(self) -> None:
        """Clear histogram cache."""
        self._cache.clear()
    
    def stats(self) -> dict:
        """Return cache statistics."""
        return self._cache.stats()


class ComputationCache:
    """
    General-purpose memoization cache for expensive computations.
    
    Supports automatic key generation from function arguments.
    """
    
    def __init__(self, max_memory_mb: float = 50.0):
        self._cache = LRUCache(max_memory_mb=max_memory_mb, max_entries=100)
    
    def _make_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Create cache key from function name and arguments."""
        h = hashlib.blake2b(digest_size=16)
        h.update(func_name.encode())
        
        # Hash args
        for arg in args:
            if isinstance(arg, np.ndarray):
                h.update(str(arg.shape).encode())
                h.update(str(arg.dtype).encode())
                if arg.size > 0:
                    # Sample for speed
                    indices = [0, arg.size // 2, arg.size - 1] if arg.size > 2 else list(range(arg.size))
                    h.update(arg.flat[indices].tobytes())
            else:
                h.update(str(arg).encode())
        
        # Hash kwargs
        for k, v in sorted(kwargs.items()):
            h.update(k.encode())
            if isinstance(v, np.ndarray):
                h.update(str(v.shape).encode())
                h.update(str(v.dtype).encode())
            else:
                h.update(str(v).encode())
        
        return h.hexdigest()
    
    def memoize(self, func: Callable) -> Callable:
        """
        Decorator to memoize function results.
        
        Example:
            @cache_manager.computation_cache.memoize
            def expensive_function(data):
                return slow_computation(data)
        """
        def wrapper(*args, **kwargs):
            key = self._make_key(func.__name__, args, kwargs)
            result = self._cache.get(key)
            if result is not None:
                logging.debug(f"[ComputationCache] Hit for {func.__name__}")
                return result
            
            logging.debug(f"[ComputationCache] Miss for {func.__name__}, computing...")
            result = func(*args, **kwargs)
            self._cache.put(key, result)
            return result
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    
    def clear(self) -> None:
        """Clear computation cache."""
        self._cache.clear()
    
    def stats(self) -> dict:
        """Return cache statistics."""
        return self._cache.stats()


class CacheManager:
    """
    Central cache manager for ndxplorer.
    
    Manages multiple specialized caches with coordinated memory limits.
    """
    
    def __init__(
        self,
        histogram_memory_mb: float = 200.0,
        computation_memory_mb: float = 50.0,
        general_memory_mb: float = 50.0,
    ):
        self.histogram_cache = HistogramCache(max_memory_mb=histogram_memory_mb)
        self.computation_cache = ComputationCache(max_memory_mb=computation_memory_mb)
        self.general_cache = LRUCache(max_memory_mb=general_memory_mb, max_entries=100)
    
    def clear_all(self) -> None:
        """Clear all caches."""
        self.histogram_cache.clear()
        self.computation_cache.clear()
        self.general_cache.clear()
        logging.info("[CacheManager] All caches cleared")
    
    def stats(self) -> dict:
        """Return statistics for all caches."""
        return {
            'histogram': self.histogram_cache.stats(),
            'computation': self.computation_cache.stats(),
            'general': self.general_cache.stats(),
        }
    
    def log_stats(self) -> None:
        """Log cache statistics."""
        stats = self.stats()
        logging.info("[CacheManager] Statistics:")
        for cache_name, cache_stats in stats.items():
            logging.info(f"  {cache_name}: {cache_stats}")


# Global cache manager instance
_global_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get or create global cache manager."""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager()
    return _global_cache_manager


def clear_all_caches() -> None:
    """Clear all ndxplorer caches."""
    manager = get_cache_manager()
    manager.clear_all()
