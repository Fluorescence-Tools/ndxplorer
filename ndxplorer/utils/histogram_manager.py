"""
Clean Histogram Manager for ndxplorer.

Simple, efficient histogram computation and caching without backward compatibility.
"""

from __future__ import annotations
import json
import hashlib
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import logging

from ..logging_config import logging
from ..core.histograms import Histogram1D, Histogram2D
from dataclasses import dataclass


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
    """Simple, efficient histogram cache."""
    
    def __init__(self, max_size_mb: float = 200.0):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._max_size_mb = max_size_mb
        self._current_size_mb = 0.0
        self._hits = 0
        self._misses = 0
    
    def _compute_cache_key(self, params: Dict[str, Any]) -> str:
        """Compute deterministic cache key from parameters."""
        # Remove None values and sort keys for consistency
        key_data = {k: v for k, v in params.items() if v is not None}
        key_str = json.dumps(key_data, sort_keys=True, separators=(',', ':'))
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _estimate_size_mb(self, hist: Union[Histogram2D, Histogram1D]) -> float:
        """Estimate memory usage in MB."""
        if isinstance(hist, Histogram2D):
            return (hist.H.nbytes + hist.x_edges.nbytes + hist.y_edges.nbytes) / (1024 * 1024)
        else:
            return (hist.counts.nbytes + hist.edges.nbytes) / (1024 * 1024)
    
    def get(self, params: Dict[str, Any], hist_type: str = '2d') -> Optional[Union[Histogram2D, Histogram1D]]:
        """Get cached histogram."""
        key = self._compute_cache_key(params)
        cache_key = f"{hist_type}_{key}"
        
        if cache_key in self._cache:
            self._hits += 1
            cached_hist = self._cache[cache_key]['histogram']
            
            # Return a copy to avoid issues with shared references
            if isinstance(cached_hist, Histogram2D):
                # Create a new Histogram2D with copied arrays
                original_H = cached_hist.H
                logging.info(f"[CACHE GET] 2D histogram BEFORE copy:")
                logging.info(f"[CACHE GET]   H shape: {original_H.shape} (expected: ({len(cached_hist.y_edges)-1}, {len(cached_hist.x_edges)-1}))")
                logging.info(f"[CACHE GET]   x_edges: {len(cached_hist.x_edges)}, y_edges: {len(cached_hist.y_edges)}")
                logging.info(f"[CACHE GET]   H[0] == y_edges-1? {original_H.shape[0] == len(cached_hist.y_edges)-1}")
                logging.info(f"[CACHE GET]   H[1] == x_edges-1? {original_H.shape[1] == len(cached_hist.x_edges)-1}")
                
                cached_hist = Histogram2D(
                    H=np.ascontiguousarray(cached_hist.H.copy()),
                    x_edges=cached_hist.x_edges.copy(),
                    y_edges=cached_hist.y_edges.copy()
                )
                
                logging.info(f"[CACHE GET] 2D histogram AFTER copy:")
                logging.info(f"[CACHE GET]   H shape: {cached_hist.H.shape}, contiguous: {cached_hist.H.flags['C_CONTIGUOUS']}")
                logging.info(f"[CACHE GET]   Validation: {cached_hist.validate()}")
            elif isinstance(cached_hist, Histogram1D):
                # Create a new Histogram1D with copied arrays
                cached_hist = Histogram1D(
                    edges=cached_hist.edges.copy(),
                    counts=cached_hist.counts.copy()
                )
            
            return cached_hist
        
        self._misses += 1
        return None
    
    def put(self, params: Dict[str, Any], histogram: Union[Histogram2D, Histogram1D]) -> None:
        """Store histogram in cache."""
        if not histogram.validate():
            logging.error(f"[CACHE PUT] Histogram validation failed, not caching")
            return
        
        hist_type = '2d' if isinstance(histogram, Histogram2D) else '1d'
        
        # Log what we're putting into cache
        if isinstance(histogram, Histogram2D):
            logging.info(f"[CACHE PUT] Storing 2D histogram:")
            logging.info(f"[CACHE PUT]   H shape: {histogram.H.shape}")
            logging.info(f"[CACHE PUT]   x_edges: {len(histogram.x_edges)}, y_edges: {len(histogram.y_edges)}")
            logging.info(f"[CACHE PUT]   Expected: H[0]={len(histogram.y_edges)-1}, H[1]={len(histogram.x_edges)-1}")
            logging.info(f"[CACHE PUT]   Actual: H[0]={histogram.H.shape[0]}, H[1]={histogram.H.shape[1]}")
            logging.info(f"[CACHE PUT]   Shape match: {histogram.H.shape == (len(histogram.y_edges)-1, len(histogram.x_edges)-1)}")
        
        key = self._compute_cache_key(params)
        cache_key = f"{hist_type}_{key}"
        
        # Check if we need to evict
        size_mb = self._estimate_size_mb(histogram)
        if size_mb > self._max_size_mb:
            return
        
        # Simple LRU: clear cache if we're over limit
        while self._current_size_mb + size_mb > self._max_size_mb and self._cache:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._cache))
            oldest_size = self._cache[oldest_key]['size_mb']
            del self._cache[oldest_key]
            self._current_size_mb -= oldest_size
        
        # Store new entry
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
    """Clean histogram computation and management."""
    
    def __init__(self, cache_size_mb: float = 200.0):
        self._cache = HistogramCache(cache_size_mb)
        self._stats = {
            'total_computations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'computation_time_ms': []
        }
    
    def compute_histogram_2d(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        x_bins: Union[int, np.ndarray],
        y_bins: Union[int, np.ndarray],
        weights: Optional[np.ndarray] = None,
        params: Optional[HistogramParams] = None
    ) -> Histogram2D:
        """Compute 2D histogram."""
        import time
        start_time = time.perf_counter()
        
        # Check cache first
        if params is not None:
            cached = self._cache.get(params.to_dict(), '2d')
            if cached is not None:
                self._stats['cache_hits'] += 1
                return cached
        
        self._stats['cache_misses'] += 1
        
        # Compute histogram
        try:
            # Handle empty or invalid data
            if len(x_data) == 0 or len(y_data) == 0:
                logging.warning("Empty data provided for 2D histogram")
                return Histogram2D(
                    H=np.zeros((1, 1)),
                    x_edges=np.array([0.0, 1.0]),
                    y_edges=np.array([0.0, 1.0])
                )
            
            # Convert bins to edges if needed
            if isinstance(x_bins, (int, np.integer)):
                x_min, x_max = np.min(x_data), np.max(x_data)
                if x_min == x_max:
                    # Handle zero range data
                    x_min -= 0.5
                    x_max += 0.5
                    logging.debug(f"Zero range in X data, expanding to [{x_min}, {x_max}]")
                x_edges = np.linspace(x_min, x_max, int(x_bins) + 1)
            else:
                x_edges = np.asarray(x_bins, dtype=np.float64)
            
            if isinstance(y_bins, (int, np.integer)):
                y_min, y_max = np.min(y_data), np.max(y_data)
                if y_min == y_max:
                    # Handle zero range data
                    y_min -= 0.5
                    y_max += 0.5
                    logging.debug(f"Zero range in Y data, expanding to [{y_min}, {y_max}]")
                y_edges = np.linspace(y_min, y_max, int(y_bins) + 1)
            else:
                y_edges = np.asarray(y_bins, dtype=np.float64)
            
            # Use numpy histogram2d
            H, x_edges_out, y_edges_out = np.histogram2d(
                x_data, y_data,
                bins=[x_edges, y_edges],
                weights=weights,
                density=False
            )
            
            # CRITICAL: np.histogram2d returns H with shape (nx, ny) but Histogram2D expects (ny, nx)
            # Must transpose to match the expected format
            logging.info(f"[IMMEDIATE] numpy histogram2d: H shape before transpose={H.shape}")
            H_transposed = H.T
            
            # Create histogram object - ensure H is C-contiguous for display
            H_clean = np.nan_to_num(H_transposed, nan=0.0, posinf=0.0, neginf=0.0)
            H_contiguous = np.ascontiguousarray(H_clean)
            histogram = Histogram2D(
                H=H_contiguous,
                x_edges=x_edges_out,
                y_edges=y_edges_out
            )
            
            logging.info(f"[IMMEDIATE] Created histogram: H shape={histogram.H.shape}, x_edges={len(histogram.x_edges)}, y_edges={len(histogram.y_edges)}")
            logging.info(f"[IMMEDIATE] H dtype={histogram.H.dtype}, contiguous={histogram.H.flags['C_CONTIGUOUS']}, min={np.min(histogram.H)}, max={np.max(histogram.H)}, sum={np.sum(histogram.H)}")
            logging.debug(f"compute_histogram_2d: Expected H shape: ({len(histogram.y_edges)-1}, {len(histogram.x_edges)-1}), Got: {histogram.H.shape}")
            
            # Validate
            if not histogram.validate():
                logging.error(f"Histogram validation FAILED!")
                logging.error(f"  H shape: {histogram.H.shape}")
                logging.error(f"  x_edges length: {len(histogram.x_edges)} (expected {histogram.H.shape[1] + 1})")
                logging.error(f"  y_edges length: {len(histogram.y_edges)} (expected {histogram.H.shape[0] + 1})")
                raise ValueError("Computed histogram failed validation")
            
            # Cache if params provided
            if params is not None:
                logging.debug(f"compute_histogram_2d: Caching histogram with shape={histogram.H.shape}")
                self._cache.put(params.to_dict(), histogram)
                logging.debug(f"compute_histogram_2d: Histogram cached successfully")
            
            # Update stats
            computation_time = (time.perf_counter() - start_time) * 1000
            self._stats['total_computations'] += 1
            self._stats['computation_time_ms'].append(computation_time)
            
            return histogram
            
        except Exception as e:
            logging.error(f"Failed to compute 2D histogram: {e}")
            # Return empty histogram
            return Histogram2D(
                H=np.zeros((1, 1)),
                x_edges=np.array([0.0, 1.0]),
                y_edges=np.array([0.0, 1.0])
            )
    
    def compute_histogram_1d(
        self,
        data: np.ndarray,
        bins: Union[int, np.ndarray],
        weights: Optional[np.ndarray] = None,
        density: bool = False,
        params: Optional[HistogramParams] = None
    ) -> Histogram1D:
        """Compute 1D histogram."""
        import time
        start_time = time.perf_counter()
        
        # Check cache first
        if params is not None:
            cached = self._cache.get(params.to_dict(), '1d')
            if cached is not None:
                self._stats['cache_hits'] += 1
                return cached
        
        self._stats['cache_misses'] += 1
        
        try:
            # Handle empty or invalid data
            if len(data) == 0:
                logging.warning("Empty data provided for 1D histogram")
                return Histogram1D(
                    edges=np.array([0.0, 1.0]),
                    counts=np.array([0.0])
                )
            
            # Convert bins to edges if needed
            if isinstance(bins, (int, np.integer)):
                data_min, data_max = np.min(data), np.max(data)
                if data_min == data_max:
                    # Handle zero range data
                    data_min -= 0.5
                    data_max += 0.5
                    logging.debug(f"Zero range in 1D data, expanding to [{data_min}, {data_max}]")
                edges = np.linspace(data_min, data_max, int(bins) + 1)
            else:
                edges = np.asarray(bins, dtype=np.float64)
            
            # Compute histogram
            counts, edges_out = np.histogram(data, bins=edges, weights=weights, density=density)
            
            # Create histogram object
            histogram = Histogram1D(
                edges=edges_out,
                counts=counts.astype(np.float64)
            )
            
            # Validate
            if not histogram.validate():
                raise ValueError("Computed histogram failed validation")
            
            # Cache if params provided
            if params is not None:
                self._cache.put(params.to_dict(), histogram)
            
            # Update stats
            computation_time = (time.perf_counter() - start_time) * 1000
            self._stats['total_computations'] += 1
            self._stats['computation_time_ms'].append(computation_time)
            
            return histogram
            
        except Exception as e:
            logging.error(f"Failed to compute 1D histogram: {e}")
            # Return empty histogram
            return Histogram1D(
                edges=np.array([0.0, 1.0]),
                counts=np.array([0])
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        cache_stats = self._cache.get_stats()
        
        avg_time_ms = np.mean(self._stats['computation_time_ms']) if self._stats['computation_time_ms'] else 0.0
        
        return {
            'cache': cache_stats,
            'computations': {
                'total': self._stats['total_computations'],
                'cache_hits': self._stats['cache_hits'],
                'cache_misses': self._stats['cache_misses'],
                'avg_time_ms': avg_time_ms,
                'total_time_ms': sum(self._stats['computation_time_ms'])
            }
        }
    
    def clear(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        self._stats = {
            'total_computations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'computation_time_ms': []
        }


# Global instance
_global_manager: Optional[HistogramManager] = None


def get_histogram_manager() -> HistogramManager:
    """Get the global histogram manager instance."""
    global _global_manager
    if _global_manager is None:
        _global_manager = HistogramManager()
    return _global_manager


def clear_histogram_manager() -> None:
    """Clear the global histogram manager."""
    global _global_manager
    if _global_manager is not None:
        _global_manager.clear()
    _global_manager = None
