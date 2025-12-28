"""Background histogram computation and caching for NDxplorer.

This module provides threaded histogram computation to keep the UI responsive
during data processing, along with enhanced caching mechanisms.
"""

from __future__ import annotations

import threading
import time
import copy
from typing import Dict, Optional, Tuple, Any, Callable
from concurrent.futures import ThreadPoolExecutor, Future
import queue

from qtpy import QtCore

import numpy as np

from ..logging_config import logging


class HistogramComputeWorker(QtCore.QObject):
    """Worker object for computing histograms in a background thread."""
    
    # Signals emitted when computation is complete or fails
    computation_complete = QtCore.Signal(dict)
    computation_failed = QtCore.Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancelled = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="histogram_worker")
        
    def compute_histograms(
        self,
        data_source,
        histogram_params: Dict[str, Any],
        weights: Optional[np.ndarray] = None
    ) -> None:
        """Compute histograms in background thread."""
        if self._cancelled:
            return
            
        try:
            future = self._executor.submit(
                self._do_compute_histograms,
                data_source,
                histogram_params,
                weights
            )
            future.add_done_callback(self._on_computation_done)
        except Exception as e:
            logging.error(f"Failed to schedule histogram computation: {e}")
            self.computation_failed.emit(str(e))
    
    def _do_compute_histograms(
        self,
        data_source,
        histogram_params: Dict[str, Any],
        weights: Optional[np.ndarray] = None
    ) -> Dict[str, Tuple]:
        """Actual histogram computation running in background thread."""
        start_time = time.time()
        
        try:
            # Extract data arrays
            x_data = data_source.values[histogram_params['x_idx'], :]
            y_data = data_source.values[histogram_params['y_idx'], :]
            z_data = data_source.values[histogram_params['z_idx'], :] if histogram_params.get('z_idx') is not None else None
            
            result = {}
            
            # Compute 1D histograms
            if not self._cancelled:
                x_hist, x_edges = np.histogram(
                    x_data, 
                    bins=histogram_params['x_bins'], 
                    range=histogram_params['x_range'],
                    weights=weights
                )
                result['x'] = (x_hist, x_edges)
            
            if not self._cancelled:
                y_hist, y_edges = np.histogram(
                    y_data,
                    bins=histogram_params['y_bins'],
                    range=histogram_params['y_range'], 
                    weights=weights
                )
                result['y'] = (y_hist, y_edges)
            
            # Compute Z histogram if available
            if not self._cancelled and z_data is not None:
                z_hist, z_edges = np.histogram(
                    z_data,
                    bins=histogram_params['z_bins'],
                    range=histogram_params['z_range'],
                    weights=weights
                )
                result['z'] = (z_hist, z_edges)
            
            # Compute 2D histogram
            if not self._cancelled:
                h2d, x_edges_2d, y_edges_2d = np.histogram2d(
                    x_data,
                    y_data,
                    bins=[histogram_params['x_bins_2d'], histogram_params['y_bins_2d']],
                    range=[histogram_params['x_range'], histogram_params['y_range']],
                    weights=weights
                )
                result['2d'] = (h2d, x_edges_2d, y_edges_2d)
            
            # Add metadata
            result['_count'] = len(x_data)
            result['_computation_time'] = time.time() - start_time
            result['_params'] = copy.deepcopy(histogram_params)
            if weights is not None:
                result['_has_weights'] = True
            
            logging.debug(f"Background histogram computation completed in {result['_computation_time']:.3f}s")
            return result
            
        except Exception as e:
            logging.error(f"Histogram computation failed: {e}")
            raise
    
    def _on_computation_done(self, future: Future) -> None:
        """Handle completion of background computation."""
        try:
            if self._cancelled:
                return
                
            result = future.result()
            if not self._cancelled:
                self.computation_complete.emit(result)
        except Exception as e:
            if not self._cancelled:
                logging.error(f"Histogram computation error: {e}")
                self.computation_failed.emit(str(e))
    
    def cancel(self) -> None:
        """Cancel any ongoing computation."""
        self._cancelled = True
        self._executor.shutdown(wait=False)
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.cancel()


class EnhancedHistogramCache:
    """Enhanced caching system for histogram data with LRU eviction and parameter-based invalidation."""
    
    def __init__(self, max_size: int = 50, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._cache: Dict[str, Dict] = {}
        self._access_times: Dict[str, float] = {}
        self._current_memory = 0
        
    def _generate_key(self, histogram_params: Dict[str, Any], weights: Optional[np.ndarray] = None) -> str:
        """Generate cache key based on histogram parameters."""
        # Create a hashable representation of parameters
        key_parts = [
            f"x_idx={histogram_params.get('x_idx')}",
            f"y_idx={histogram_params.get('y_idx')}",
            f"z_idx={histogram_params.get('z_idx')}",
            f"x_bins={histogram_params.get('x_bins')}",
            f"y_bins={histogram_params.get('y_bins')}",
            f"z_bins={histogram_params.get('z_bins')}",
            f"x_bins_2d={histogram_params.get('x_bins_2d')}",
            f"y_bins_2d={histogram_params.get('y_bins_2d')}",
            f"x_range={tuple(histogram_params.get('x_range', (0, 1)))}",
            f"y_range={tuple(histogram_params.get('y_range', (0, 1)))}",
            f"z_range={tuple(histogram_params.get('z_range', (0, 1)))}",
        ]
        
        # Add weight information
        if weights is not None:
            key_parts.append(f"weights_hash={hash(weights.tobytes())}")
        
        return "|".join(key_parts)
    
    def get(self, histogram_params: Dict[str, Any], weights: Optional[np.ndarray] = None) -> Optional[Dict]:
        """Get cached histogram data if available and valid."""
        key = self._generate_key(histogram_params, weights)
        
        if key not in self._cache:
            return None
        
        # Update access time
        self._access_times[key] = time.time()
        
        # Return a deep copy to avoid modification issues
        return copy.deepcopy(self._cache[key])
    
    def put(self, histogram_params: Dict[str, Any], histogram_data: Dict, weights: Optional[np.ndarray] = None) -> None:
        """Cache histogram data with memory management."""
        key = self._generate_key(histogram_params, weights)
        
        # Estimate memory usage (rough approximation)
        data_size = sum(
            arr.nbytes if isinstance(arr, np.ndarray) else len(str(arr)) * 8
            for arr in histogram_data.values()
            if isinstance(arr, (np.ndarray, str, int, float))
        )
        
        # Evict old entries if necessary
        while (len(self._cache) >= self.max_size or 
               self._current_memory + data_size > self.max_memory_bytes) and self._cache:
            self._evict_lru()
        
        # Store the data
        self._cache[key] = copy.deepcopy(histogram_data)
        self._access_times[key] = time.time()
        self._current_memory += data_size
        
        logging.debug(f"Cached histogram (key: {key[:50]}...). Cache size: {len(self._cache)}, Memory: {self._current_memory / 1024 / 1024:.1f}MB")
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry from cache."""
        if not self._cache:
            return
        
        # Find the least recently used key
        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        
        # Remove from cache
        if lru_key in self._cache:
            del self._cache[lru_key]
            del self._access_times[lru_key]
            logging.debug(f"Evicted LRU cache entry: {lru_key[:50]}...")
    
    def invalidate_by_parameter(self, param_idx: int) -> None:
        """Invalidate cache entries that depend on a specific parameter."""
        keys_to_remove = []
        
        for key in self._cache.keys():
            if f"x_idx={param_idx}" in key or f"y_idx={param_idx}" in key or f"z_idx={param_idx}" in key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            if key in self._cache:
                del self._cache[key]
                del self._access_times[key]
        
        if keys_to_remove:
            logging.debug(f"Invalidated {len(keys_to_remove)} cache entries for parameter {param_idx}")
    
    def clear(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        self._access_times.clear()
        self._current_memory = 0
        logging.debug("Cleared histogram cache")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'size': len(self._cache),
            'memory_bytes': self._current_memory,
            'memory_mb': self._current_memory / 1024 / 1024,
            'max_size': self.max_size,
            'max_memory_mb': self.max_memory_bytes / 1024 / 1024
        }
