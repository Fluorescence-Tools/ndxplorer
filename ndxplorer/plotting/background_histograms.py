"""Background histogram computation and caching for NDxplorer.

This module provides threaded histogram computation to keep the UI responsive
during data processing, along with enhanced caching mechanisms.
"""

from __future__ import annotations

import time
import copy
from typing import Dict, Optional, Tuple, Any

from qtpy import QtCore

import numpy as np

from ..logging_config import logging


class HistogramComputationManager(QtCore.QObject):
    """Manager for computing histograms in a background thread."""
    
    # Signals emitted when computation is complete or fails
    computation_complete = QtCore.Signal(dict)
    computation_failed = QtCore.Signal(str)
    computation_started = QtCore.Signal()
    progress_update = QtCore.Signal(str)  # Progress message
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancelled = False
        self._thread = None
        self._worker = None
        
    def compute_histograms(
        self,
        data_source,
        histogram_params: Dict[str, Any],
        weights: Optional[np.ndarray] = None
    ) -> None:
        """Compute histograms in background thread using QThread."""
        logging.debug("[BG WORKER] compute_histograms called")
        if self._cancelled:
            logging.debug("[BG WORKER] Worker cancelled, skipping")
            return
            
        # Clean up any existing thread
        if self._thread and self._thread.isRunning():
            logging.debug("[BG WORKER] Cleaning up existing thread")
            self._thread.quit()
            self._thread.wait()
            
        # Create new thread and worker
        logging.debug("[BG WORKER] Creating new thread and worker")
        self._thread = QtCore.QThread()
        self._worker = HistogramComputationWorker(
            data_source, histogram_params, weights
        )
        self._worker.moveToThread(self._thread)
        
        # Connect signals
        logging.debug("[BG WORKER] Connecting signals")
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.error.connect(self._on_worker_error)
        self._worker.progress.connect(self._on_worker_progress)
        self._thread.started.connect(self._worker.run)
        
        # Emit started signal
        logging.debug("[BG WORKER] Emitting started signal")
        self.computation_started.emit()
        
        # Start the thread
        logging.debug("[BG WORKER] Starting thread")
        self._thread.start()
        logging.debug("[BG WORKER] Thread started")
    
    def _on_worker_finished(self, result):
        """Handle worker completion."""
        logging.debug(f"[BG MANAGER] _on_worker_finished called with {len(result)} result items")
        if self._thread:
            self._thread.quit()
            self._thread.wait()
        if not self._cancelled:
            logging.debug("[BG MANAGER] Emitting computation_complete signal")
            self.computation_complete.emit(result)
        else:
            logging.debug("[BG MANAGER] Cancelled, not emitting completion signal")
    
    def _on_worker_error(self, error_msg):
        """Handle worker error."""
        if self._thread:
            self._thread.quit()
            self._thread.wait()
        if not self._cancelled:
            self.computation_failed.emit(error_msg)
    
    def _on_worker_progress(self, message):
        """Handle worker progress update."""
        if not self._cancelled:
            self.progress_update.emit(message)
    
    def cancel(self) -> None:
        """Cancel any ongoing computation."""
        self._cancelled = True
        if self._worker:
            self._worker.cancel()
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.cancel()


class HistogramComputationWorker(QtCore.QObject):
    """Actual computation worker that runs in background thread."""
    
    finished = QtCore.Signal(dict)
    error = QtCore.Signal(str)
    progress = QtCore.Signal(str)
    
    def __init__(self, data_source, histogram_params: Dict[str, Any], weights: Optional[np.ndarray] = None):
        super().__init__()
        self.data_source = data_source
        self.histogram_params = histogram_params
        self.weights = weights
        self._cancelled = False
    
    def run(self):
        """Run the computation."""
        logging.debug("[BG WORKER RUN] Worker run() called")
        try:
            logging.debug("[BG WORKER RUN] Starting histogram computation")
            result = self._do_compute_histograms()
            logging.debug(f"[BG WORKER RUN] Computation complete, result keys: {list(result.keys())}")
            if not self._cancelled:
                logging.debug("[BG WORKER RUN] Emitting finished signal")
                self.finished.emit(result)
            else:
                logging.debug("[BG WORKER RUN] Worker was cancelled, not emitting")
        except Exception as e:
            logging.error(f"[BG WORKER RUN] Error during computation: {e}")
            import traceback
            logging.error(f"[BG WORKER RUN] Traceback:\n{traceback.format_exc()}")
            if not self._cancelled:
                self.error.emit(str(e))
    
    def cancel(self):
        """Cancel the computation."""
        self._cancelled = True
    
    def _do_compute_histograms(
        self,
    ) -> Dict[str, Tuple]:
        """Actual histogram computation running in background thread."""
        start_time = time.time()
        
        # Emit initial progress
        self.progress.emit("Computing histograms...")
        
        try:
            # Extract data arrays
            self.progress.emit("Extracting data arrays...")
            x_data = self.data_source.values[self.histogram_params['x_idx'], :]
            y_data = self.data_source.values[self.histogram_params['y_idx'], :]
            z_data = (
                self.data_source.values[self.histogram_params['z_idx'], :]
                if self.histogram_params.get('z_idx') is not None
                else None
            )

            valid_indices = self.histogram_params.get('valid_indices')
            if valid_indices is not None:
                try:
                    x_data = x_data[valid_indices]
                    y_data = y_data[valid_indices]
                    if z_data is not None:
                        z_data = z_data[valid_indices]
                    if self.weights is not None:
                        self.weights = self.weights[valid_indices]
                except Exception as exc:
                    logging.debug("Failed to apply valid_indices filter in background histograms: %s", exc)
            
            result = {}
            
            # Check performance configuration for histogram backend
            from ..utils.performance_config import get_performance_config
            config = get_performance_config()
            
            if config.use_boost_histogram:
                # Use boost-histogram for maximum performance
                try:
                    import boost_histogram as bh
                    
                    # Compute 1D histograms using boost-histogram
                    if not self._cancelled:
                        self.progress.emit("Computing X histogram...")
                        # Use actual bin arrays if available, otherwise fall back to integer count
                        x_bins_param = self.histogram_params.get('x_bins_1d_arr', self.histogram_params['x_bins'])
                        density_x = self.histogram_params.get('normed_x', False)
                        logging.debug(f"[BG] Computing X histogram with boost-histogram, density={density_x}")
                        
                        if hasattr(x_bins_param, '__len__') and len(x_bins_param) > 1:
                            # Use actual bin edges
                            x_hist = bh.Histogram(bh.axis.Variable(x_bins_param))
                        else:
                            # Use integer count with range
                            x_hist = bh.Histogram(bh.axis.Regular(x_bins_param, *self.histogram_params['x_range']))
                        
                        if self.weights is not None:
                            x_hist.fill(x_data, weight=self.weights)
                        else:
                            x_hist.fill(x_data)
                        
                        x_values = x_hist.values()
                        if density_x:
                            # Apply density normalization manually
                            bin_widths = np.diff(x_hist.axes[0].edges)
                            total = np.sum(x_values * bin_widths)
                            if total > 0:
                                x_values = x_values / (total * bin_widths)
                            logging.debug(f"[BG] Applied density normalization to X histogram")
                        
                        result['x'] = (x_hist.axes[0].edges, x_values)
                    
                    if not self._cancelled:
                        self.progress.emit("Computing Y histogram...")
                        # Use actual bin arrays if available, otherwise fall back to integer count
                        y_bins_param = self.histogram_params.get('y_bins_1d_arr', self.histogram_params['y_bins'])
                        density_y = self.histogram_params.get('normed_y', False)
                        logging.debug(f"[BG] Computing Y histogram with boost-histogram, density={density_y}")
                        
                        if hasattr(y_bins_param, '__len__') and len(y_bins_param) > 1:
                            # Use actual bin edges
                            y_hist = bh.Histogram(bh.axis.Variable(y_bins_param))
                        else:
                            # Use integer count with range
                            y_hist = bh.Histogram(bh.axis.Regular(y_bins_param, *self.histogram_params['y_range']))
                        
                        if self.weights is not None:
                            y_hist.fill(y_data, weight=self.weights)
                        else:
                            y_hist.fill(y_data)
                        
                        y_values = y_hist.values()
                        if density_y:
                            # Apply density normalization manually
                            bin_widths = np.diff(y_hist.axes[0].edges)
                            total = np.sum(y_values * bin_widths)
                            if total > 0:
                                y_values = y_values / (total * bin_widths)
                            logging.debug(f"[BG] Applied density normalization to Y histogram")
                        
                        result['y'] = (y_hist.axes[0].edges, y_values)
                    
                    # Compute Z histogram if available
                    if not self._cancelled and z_data is not None:
                        # Use actual bin arrays if available, otherwise fall back to integer count
                        z_bins_param = self.histogram_params.get('z_bins_1d_arr', self.histogram_params['z_bins'])
                        density_z = self.histogram_params.get('normed_z', False)
                        logging.debug(f"[BG] Computing Z histogram with boost-histogram, density={density_z}")
                        
                        if hasattr(z_bins_param, '__len__') and len(z_bins_param) > 1:
                            # Use actual bin edges
                            z_hist = bh.Histogram(bh.axis.Variable(z_bins_param))
                        else:
                            # Use integer count with range
                            z_hist = bh.Histogram(bh.axis.Regular(z_bins_param, *self.histogram_params['z_range']))
                        
                        if self.weights is not None:
                            z_hist.fill(z_data, weight=self.weights)
                        else:
                            z_hist.fill(z_data)
                        
                        z_values = z_hist.values()
                        if density_z:
                            # Apply density normalization manually
                            bin_widths = np.diff(z_hist.axes[0].edges)
                            total = np.sum(z_values * bin_widths)
                            if total > 0:
                                z_values = z_values / (total * bin_widths)
                            logging.debug(f"[BG] Applied density normalization to Z histogram")
                        
                        result['z'] = (z_hist.axes[0].edges, z_values)
                    
                    # Compute 2D histogram
                    if not self._cancelled:
                        self.progress.emit("Computing 2D histogram...")
                        # Use actual bin arrays if available, otherwise fall back to integer count
                        x_bins_2d_param = self.histogram_params.get('x_bins_2d_arr', self.histogram_params['x_bins_2d'])
                        y_bins_2d_param = self.histogram_params.get('y_bins_2d_arr', self.histogram_params['y_bins_2d'])
                        
                        if (hasattr(x_bins_2d_param, '__len__') and len(x_bins_2d_param) > 1 and 
                            hasattr(y_bins_2d_param, '__len__') and len(y_bins_2d_param) > 1):
                            # Use actual bin edges
                            h2d = bh.Histogram(
                                bh.axis.Variable(x_bins_2d_param),
                                bh.axis.Variable(y_bins_2d_param)
                            )
                        else:
                            # Use integer counts with ranges
                            h2d = bh.Histogram(
                                bh.axis.Regular(x_bins_2d_param, *self.histogram_params['x_range']),
                                bh.axis.Regular(y_bins_2d_param, *self.histogram_params['y_range'])
                            )
                        
                        if self.weights is not None:
                            h2d.fill(x_data, y_data, weight=self.weights)
                        else:
                            h2d.fill(x_data, y_data)
                        # CRITICAL: boost-histogram returns H with shape (nx, ny) but Histogram2D expects (ny, nx)
                        H_boost = h2d.values()
                        logging.debug(f"[BG] boost-histogram 2D: H shape before transpose={H_boost.shape}")
                        result['2d'] = (H_boost.T, h2d.axes[0].edges, h2d.axes[1].edges)
                        logging.debug(f"[BG] boost-histogram 2D: H shape after transpose={H_boost.T.shape}, x_edges={len(h2d.axes[0].edges)}, y_edges={len(h2d.axes[1].edges)}")
                        
                except ImportError:
                    logging.warning("boost-histogram not available, falling back to numpy")
                    config.use_boost_histogram = False
            
            if not config.use_boost_histogram:
                # Use numpy histogram (fallback/default)
                logging.debug("Using numpy histogram computation in background")
                
                # Compute 1D histograms using numpy
                if not self._cancelled:
                    # Use actual bin arrays if available, otherwise fall back to integer count
                    x_bins_param = self.histogram_params.get('x_bins_1d_arr', self.histogram_params['x_bins'])
                    density_x = self.histogram_params.get('normed_x', False)
                    logging.debug(f"[BG] Computing X histogram with density={density_x}")
                    x_hist, x_edges = np.histogram(x_data, bins=x_bins_param, range=self.histogram_params['x_range'], weights=self.weights, density=density_x)
                    result['x'] = (x_edges, x_hist)  # Store as (edges, counts)
                
                if not self._cancelled:
                    # Use actual bin arrays if available, otherwise fall back to integer count
                    y_bins_param = self.histogram_params.get('y_bins_1d_arr', self.histogram_params['y_bins'])
                    density_y = self.histogram_params.get('normed_y', False)
                    logging.debug(f"[BG] Computing Y histogram with density={density_y}")
                    y_hist, y_edges = np.histogram(y_data, bins=y_bins_param, range=self.histogram_params['y_range'], weights=self.weights, density=density_y)
                    result['y'] = (y_edges, y_hist)  # Store as (edges, counts)
                
                # Compute Z histogram if available
                if not self._cancelled and z_data is not None:
                    # Use actual bin arrays if available, otherwise fall back to integer count
                    z_bins_param = self.histogram_params.get('z_bins_1d_arr', self.histogram_params['z_bins'])
                    density_z = self.histogram_params.get('normed_z', False)
                    logging.debug(f"[BG] Computing Z histogram with density={density_z}")
                    z_hist, z_edges = np.histogram(z_data, bins=z_bins_param, range=self.histogram_params['z_range'], weights=self.weights, density=density_z)
                    result['z'] = (z_edges, z_hist)  # Store as (edges, counts)
                
                # Compute 2D histogram using numpy
                if not self._cancelled:
                    # Use actual bin arrays if available, otherwise fall back to integer count
                    x_bins_2d_param = self.histogram_params.get('x_bins_2d_arr', self.histogram_params['x_bins_2d'])
                    y_bins_2d_param = self.histogram_params.get('y_bins_2d_arr', self.histogram_params['y_bins_2d'])
                    
                    H, x_edges, y_edges = np.histogram2d(
                        x_data, y_data, 
                        bins=[x_bins_2d_param, y_bins_2d_param],
                        range=[self.histogram_params['x_range'], self.histogram_params['y_range']],
                        weights=self.weights
                    )
                    # CRITICAL: np.histogram2d returns H with shape (nx, ny) but Histogram2D expects (ny, nx)
                    logging.debug(f"[BG] numpy histogram2d: H shape before transpose={H.shape}")
                    result['2d'] = (H.T, x_edges, y_edges)
                    logging.debug(f"[BG] numpy histogram2d: H shape after transpose={H.T.shape}, x_edges={len(x_edges)}, y_edges={len(y_edges)}")
            
            # Add metadata
            result['_count'] = len(x_data)
            result['_computation_time'] = time.time() - start_time
            params_copy = copy.deepcopy(self.histogram_params)
            # Do not persist large index arrays inside cache metadata
            params_copy.pop('valid_indices', None)
            result['_params'] = params_copy
            if self.weights is not None:
                result['_has_weights'] = True
            
            # Final shape validation before returning
            if '2d' in result:
                H, x_e, y_e = result['2d']
                logging.debug(f"[BG FINAL] Returning 2D histogram: H shape={H.shape}, x_edges={len(x_e)}, y_edges={len(y_e)}")
            
            logging.debug(f"Background histogram computation completed in {result['_computation_time']:.3f}s")
            return result
            
        except Exception as e:
            logging.error(f"Histogram computation failed: {e}")
            raise


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
            f"valid_idx_hash={histogram_params.get('valid_idx_hash')}",
            f"valid_idx_count={histogram_params.get('valid_idx_count')}",
            f"selection_hash={histogram_params.get('selection_hash')}",
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
        cached_data = copy.deepcopy(self._cache[key])
        
        # Convert 2D histogram tuple to Histogram2D object if needed
        if '2d' in cached_data and isinstance(cached_data['2d'], tuple) and len(cached_data['2d']) == 3:
            from ..core.histograms import Histogram2D
            H, x_edges, y_edges = cached_data['2d']
            cached_data['2d'] = Histogram2D(
                H=np.ascontiguousarray(H),
                x_edges=x_edges,
                y_edges=y_edges
            )
            logging.debug("[CACHE GET] Converted 2D histogram tuple to Histogram2D object")
        
        return cached_data
    
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
