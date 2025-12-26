"""
PyArrow-based data backend for NDXplorer.

This module provides a high-performance data backend using Apache Arrow
for columnar data storage and operations. Key benefits:
- 2-5x faster CSV reading
- Zero-copy numpy conversions
- Better memory efficiency
- Fast compute kernels for filtering/aggregation

The ArrowDataSource class is a drop-in replacement for DataSource when
PyArrow is available, falling back gracefully when it's not.
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional, Tuple, Set, Any

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pa_csv
    HAVE_PYARROW = True
except ImportError:
    pa = None
    pc = None
    pa_csv = None
    HAVE_PYARROW = False

try:
    import pandas as pd
    HAVE_PANDAS = True
except ImportError:
    pd = None
    HAVE_PANDAS = False

from ..logging_config import logging

# Optional Numba acceleration (imported from data_source)
try:
    from .data_source import (
        _HAVE_NUMBA,
        _rectangular_mask_numba,
        _gaussian2d_mask_numba,
        _mask_nan_inf_numba,
        DataSelection,
        RectangularDataSelection,
        Gaussian2DSelection,
    )
except ImportError:
    _HAVE_NUMBA = False
    _rectangular_mask_numba = None
    _gaussian2d_mask_numba = None
    _mask_nan_inf_numba = None
    DataSelection = None
    RectangularDataSelection = None
    Gaussian2DSelection = None


def is_pyarrow_available() -> bool:
    """Check if PyArrow is available."""
    return HAVE_PYARROW


class ArrowDataSource:
    """
    High-performance data source using Apache Arrow.
    
    This class provides the same interface as DataSource but uses Arrow Tables
    internally for better performance on large datasets.
    
    Key optimizations:
    - Zero-copy conversion to numpy arrays
    - Lazy evaluation of numeric conversion
    - Efficient columnar storage
    - Fast filtering with Arrow compute kernels
    """
    
    _table: Optional["pa.Table"]
    _parameter_names: List[str]
    _cached_values_array: Optional[np.ndarray]
    _column_arrays: Dict[str, np.ndarray]
    
    def __init__(
        self,
        parameter_names: Optional[List[str]] = None,
        data: Optional[Any] = None,
    ):
        if not HAVE_PYARROW:
            raise ImportError("PyArrow is required for ArrowDataSource")
        
        self._table = None
        self._parameter_names = []
        self._cached_values_array = None
        self._column_arrays = {}
        self._relevant_columns_cache = None
        
        if data is not None:
            self.data = data
        elif parameter_names:
            self._parameter_names = list(parameter_names)
    
    def __str__(self) -> str:
        if self._table is None:
            return "ArrowDataSource(empty)"
        return f"ArrowDataSource({self._table.num_rows} rows, {self._table.num_columns} columns)"
    
    def __len__(self) -> int:
        return self.size
    
    # ---- Data loading ----
    
    @classmethod
    def from_csv(
        cls,
        path: str,
        **kwargs,
    ) -> "ArrowDataSource":
        """
        Load data from CSV file using PyArrow's fast CSV reader.
        
        This is significantly faster than pandas for large files.
        """
        if not HAVE_PYARROW:
            raise ImportError("PyArrow is required")
        
        import time
        t0 = time.perf_counter()
        
        # Configure PyArrow CSV reader for maximum speed
        read_options = pa_csv.ReadOptions(
            use_threads=True,
            block_size=1024 * 1024 * 16,  # 16MB blocks for parallel reading
        )
        
        parse_options = pa_csv.ParseOptions(
            delimiter=kwargs.get('delimiter', ','),
            quote_char=kwargs.get('quotechar', '"'),
        )
        
        convert_options = pa_csv.ConvertOptions(
            strings_can_be_null=True,
            include_columns=kwargs.get('usecols'),
            auto_dict_encode=False,  # Disable dictionary encoding for numeric data
        )
        
        try:
            table = pa_csv.read_csv(
                path,
                read_options=read_options,
                parse_options=parse_options,
                convert_options=convert_options,
            )
        except Exception as e:
            logging.warning("PyArrow CSV read failed: %s. Falling back to pandas.", e)
            if HAVE_PANDAS:
                df = pd.read_csv(path, **kwargs)
                return cls.from_pandas(df)
            raise
        
        t1 = time.perf_counter()
        logging.info("[ArrowDataSource.from_csv] Read %d rows in %.2fs", table.num_rows, t1 - t0)
        
        instance = cls()
        instance._set_table(table)
        return instance
    
    @classmethod
    def from_pandas(cls, df: "pd.DataFrame") -> "ArrowDataSource":
        """Convert pandas DataFrame to ArrowDataSource."""
        if not HAVE_PYARROW:
            raise ImportError("PyArrow is required")
        
        import time
        t0 = time.perf_counter()
        
        # Convert to Arrow Table
        table = pa.Table.from_pandas(df, preserve_index=False)
        
        t1 = time.perf_counter()
        logging.info("[ArrowDataSource.from_pandas] Converted %d rows in %.3fs", len(df), t1 - t0)
        
        instance = cls()
        instance._set_table(table)
        return instance
    
    @classmethod
    def from_numpy(
        cls,
        data: np.ndarray,
        column_names: Optional[List[str]] = None,
    ) -> "ArrowDataSource":
        """Create from numpy array (rows x columns)."""
        if not HAVE_PYARROW:
            raise ImportError("PyArrow is required")
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        n_cols = data.shape[1]
        if column_names is None:
            column_names = [f"col_{i}" for i in range(n_cols)]
        
        # Create Arrow arrays for each column
        arrays = [pa.array(data[:, i]) for i in range(n_cols)]
        table = pa.Table.from_arrays(arrays, names=column_names)
        
        instance = cls()
        instance._set_table(table)
        return instance
    
    def _set_table(self, table: "pa.Table") -> None:
        """Set the internal Arrow table and invalidate caches."""
        self._table = table
        self._parameter_names = list(table.column_names)
        self._cached_values_array = None
        self._column_arrays.clear()
        self._relevant_columns_cache = None
    
    # ---- Properties ----
    
    @property
    def parameter_names(self) -> List[str]:
        return self._parameter_names
    
    @property
    def empty(self) -> bool:
        return self._table is None or self._table.num_rows == 0
    
    @property
    def size(self) -> int:
        if self._table is None:
            return 0
        return self._table.num_rows
    
    @property
    def values(self) -> np.ndarray:
        """
        Returns (n_parameters, n_points) numeric np.ndarray.
        
        Uses zero-copy conversion when possible, with float32 for memory efficiency.
        """
        if self._cached_values_array is not None:
            return self._cached_values_array
        
        if self._table is None:
            return np.empty((0, 0), dtype=np.float32)
        
        import time
        t0 = time.perf_counter()
        
        n_cols = self._table.num_columns
        n_rows = self._table.num_rows
        
        # Pre-allocate array
        result = np.empty((n_cols, n_rows), dtype=np.float32)
        
        for i, col_name in enumerate(self._parameter_names):
            col = self._table.column(col_name)
            # Convert to numpy, coercing to float32
            try:
                # Try zero-copy first
                arr = col.to_numpy(zero_copy_only=False)
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32, copy=False)
                result[i, :] = arr
            except Exception:
                # Fallback: convert via pandas
                arr = col.to_pandas().values.astype(np.float32)
                result[i, :] = arr
        
        t1 = time.perf_counter()
        logging.debug("[ArrowDataSource.values] Converted %d×%d in %.3fs", n_cols, n_rows, t1 - t0)
        
        self._cached_values_array = result
        return result
    
    def get_column(self, name_or_idx: Any) -> np.ndarray:
        """Get a single column as numpy array (cached)."""
        if isinstance(name_or_idx, int):
            if name_or_idx < 0 or name_or_idx >= len(self._parameter_names):
                raise IndexError(f"Column index {name_or_idx} out of range")
            name = self._parameter_names[name_or_idx]
        else:
            name = str(name_or_idx)
        
        if name in self._column_arrays:
            return self._column_arrays[name]
        
        if self._table is None:
            return np.empty(0, dtype=np.float32)
        
        col = self._table.column(name)
        arr = col.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        self._column_arrays[name] = arr
        return arr
    
    # ---- DataFrame compatibility ----
    
    @property
    def data(self) -> "pd.DataFrame":
        """Return data as pandas DataFrame for compatibility."""
        if not HAVE_PANDAS:
            raise ImportError("pandas required for DataFrame conversion")
        if self._table is None:
            return pd.DataFrame()
        return self._table.to_pandas()
    
    @data.setter
    def data(self, v: Any) -> None:
        """Set data from various sources."""
        self._cached_values_array = None
        self._column_arrays.clear()
        self._relevant_columns_cache = None
        
        if v is None:
            self._table = None
            self._parameter_names = []
        elif HAVE_PYARROW and isinstance(v, pa.Table):
            self._set_table(v)
        elif HAVE_PANDAS and isinstance(v, pd.DataFrame):
            if v.empty:
                self._table = None
                self._parameter_names = []
            else:
                table = pa.Table.from_pandas(v, preserve_index=False)
                self._set_table(table)
        elif isinstance(v, np.ndarray):
            if v.size == 0:
                self._table = None
                self._parameter_names = []
            else:
                if v.ndim == 1:
                    v = v.reshape(-1, 1)
                n_cols = v.shape[1]
                names = [f"col_{i}" for i in range(n_cols)]
                arrays = [pa.array(v[:, i]) for i in range(n_cols)]
                table = pa.Table.from_arrays(arrays, names=names)
                self._set_table(table)
        else:
            raise TypeError(f"Cannot set data from type {type(v)}")
    
    def clear(self) -> None:
        """Clear all data."""
        self._table = None
        self._parameter_names = []
        self._cached_values_array = None
        self._column_arrays.clear()
        self._relevant_columns_cache = None
    
    # ---- Compute columns ----
    
    def compute_columns(
        self,
        constants: Dict[str, float],
        equations: Optional[List[Dict[str, str]]] = None,
        equation_json_fn: Optional[str] = None,
        engine: str = "python",
    ) -> None:
        """
        Compute new columns from equations.
        
        For complex equations, falls back to pandas-based computation.
        """
        if not equations and not equation_json_fn:
            return
        
        if self._table is None or self._table.num_rows == 0:
            return
        
        # For equation evaluation, convert to pandas temporarily
        # (Arrow compute doesn't support arbitrary expressions yet)
        if HAVE_PANDAS:
            from .data_source import compute_values
            df = self._table.to_pandas()
            compute_values(df, constants, equations, equation_json_fn, engine)
            # Convert back to Arrow
            self._set_table(pa.Table.from_pandas(df, preserve_index=False))
    
    # ---- Masking ----
    
    def get_mask(
        self,
        selections: List["DataSelection"],
        idxs: Optional[List[int]] = None,
        mask_nan: bool = True,
        mask_inf: bool = True,
    ) -> np.ndarray:
        """
        Combine selection masks and optionally mask NaN/Inf.
        
        Returns
        -------
        mask : np.ndarray (bool), shape (n_parameters, n_points)
            True → masked/excluded.
        """
        if DataSelection is None:
            # Fall back to simple implementation
            return self._get_mask_simple(idxs, mask_nan, mask_inf)
        
        idxs = idxs or []
        d = self.values
        n_param, n_pts = d.shape
        
        mask = np.zeros((n_param, n_pts), dtype=bool)
        
        if not selections and not idxs:
            return mask
        
        for sel in selections:
            try:
                m = sel.get_mask(d)
                if isinstance(m, np.ndarray) and m.shape == mask.shape:
                    mask |= m
            except Exception as e:
                print(f"[ArrowDataSource.get_mask] Selection error: {e}", file=sys.stderr)
        
        if idxs:
            valid_idxs = np.array([idx for idx in idxs if 0 <= idx < n_param], dtype=int)
            if valid_idxs.size:
                cols = d[valid_idxs, :]
                bad_mask = np.zeros(n_pts, dtype=bool)
                if mask_nan:
                    bad_mask |= np.any(np.isnan(cols), axis=0)
                if mask_inf:
                    bad_mask |= np.any(np.isinf(cols), axis=0)
                if np.any(bad_mask):
                    mask[:, bad_mask] = True
        
        return mask
    
    def _get_mask_simple(
        self,
        idxs: Optional[List[int]],
        mask_nan: bool,
        mask_inf: bool,
    ) -> np.ndarray:
        """Simple mask computation without selection support."""
        d = self.values
        n_param, n_pts = d.shape
        mask = np.zeros((n_param, n_pts), dtype=bool)
        
        if not idxs:
            return mask
        
        valid_idxs = np.array([idx for idx in idxs if 0 <= idx < n_param], dtype=int)
        if valid_idxs.size:
            cols = d[valid_idxs, :]
            bad_mask = np.zeros(n_pts, dtype=bool)
            if mask_nan:
                bad_mask |= np.any(np.isnan(cols), axis=0)
            if mask_inf:
                bad_mask |= np.any(np.isinf(cols), axis=0)
            if np.any(bad_mask):
                mask[:, bad_mask] = True
        
        return mask
    
    def get_relevant_column_indices(
        self,
        axis_indices: List[int],
        selections: List["DataSelection"],
        extra_indices: Optional[List[int]] = None,
    ) -> List[int]:
        """Collect column indices needed for current operations."""
        indices: Set[int] = set(axis_indices)
        if extra_indices:
            indices.update(extra_indices)
        
        if RectangularDataSelection is not None:
            for sel in selections:
                if isinstance(sel, RectangularDataSelection):
                    indices.add(sel.parameter_idx)
                elif Gaussian2DSelection is not None and isinstance(sel, Gaussian2DSelection):
                    indices.add(sel.parameter_idx1)
                    indices.add(sel.parameter_idx2)
        
        n_cols = len(self._parameter_names)
        return sorted(idx for idx in indices if 0 <= idx < n_cols)
    
    def get_values_subset(
        self,
        column_indices: List[int],
    ) -> Tuple[np.ndarray, Dict[int, int]]:
        """Return subset of values array for specified columns."""
        cache_key = tuple(column_indices)
        if (
            self._relevant_columns_cache is not None
            and self._relevant_columns_cache[0] == cache_key
        ):
            return self._relevant_columns_cache[1], self._relevant_columns_cache[2]
        
        all_values = self.values
        if not column_indices:
            empty = np.empty((0, all_values.shape[1]), dtype=np.float32)
            return empty, {}
        
        subset = all_values[column_indices, :]
        index_map = {orig: new for new, orig in enumerate(column_indices)}
        self._relevant_columns_cache = (cache_key, subset, index_map)
        return subset, index_map
    
    def get_mask_subset(
        self,
        selections: List["DataSelection"],
        axis_indices: List[int],
        mask_nan: bool = True,
        mask_inf: bool = True,
    ) -> np.ndarray:
        """Compute mask using only relevant columns."""
        relevant_indices = self.get_relevant_column_indices(axis_indices, selections)
        if not relevant_indices:
            return np.zeros(self.size, dtype=bool)
        
        subset, index_map = self.get_values_subset(relevant_indices)
        n_pts = subset.shape[1]
        mask = np.zeros(n_pts, dtype=bool)
        
        if RectangularDataSelection is None:
            return mask
        
        for sel in selections:
            if not getattr(sel, 'enabled', True):
                continue
            try:
                if isinstance(sel, RectangularDataSelection):
                    new_idx = index_map.get(sel.parameter_idx)
                    if new_idx is None:
                        continue
                    vals = np.ascontiguousarray(subset[new_idx, :], dtype=np.float64)
                    
                    if _HAVE_NUMBA and _rectangular_mask_numba is not None:
                        _rectangular_mask_numba(vals, sel.lower, sel.upper, sel.invert, mask)
                    else:
                        if sel.invert:
                            mask |= (vals > sel.lower) & (vals < sel.upper)
                        else:
                            mask |= (vals < sel.lower) | (vals > sel.upper)
                
                elif Gaussian2DSelection is not None and isinstance(sel, Gaussian2DSelection):
                    new_idx1 = index_map.get(sel.parameter_idx1)
                    new_idx2 = index_map.get(sel.parameter_idx2)
                    if new_idx1 is None or new_idx2 is None:
                        continue
                    x = np.ascontiguousarray(subset[new_idx1, :], dtype=np.float64)
                    y = np.ascontiguousarray(subset[new_idx2, :], dtype=np.float64)
                    
                    try:
                        inv_cov = np.linalg.inv(sel.cov)
                    except Exception:
                        inv_cov = np.linalg.pinv(sel.cov)
                    
                    if _HAVE_NUMBA and _gaussian2d_mask_numba is not None:
                        _gaussian2d_mask_numba(
                            x, y,
                            float(sel.mu[0]), float(sel.mu[1]),
                            float(inv_cov[0, 0]), float(inv_cov[0, 1]), float(inv_cov[1, 1]),
                            float(sel.sigma * sel.sigma),
                            sel.invert, sel.log_x, sel.log_y,
                            mask
                        )
                    else:
                        with np.errstate(divide='ignore', invalid='ignore'):
                            zx = np.where(x > 0.0, np.log(x), np.nan) if sel.log_x else x
                            zy = np.where(y > 0.0, np.log(y), np.nan) if sel.log_y else y
                        dx = zx - sel.mu[0]
                        dy = zy - sel.mu[1]
                        invalid = ~np.isfinite(dx) | ~np.isfinite(dy)
                        dx = np.nan_to_num(dx, nan=np.inf)
                        dy = np.nan_to_num(dy, nan=np.inf)
                        a, b, c = inv_cov[0, 0], inv_cov[0, 1], inv_cov[1, 1]
                        d2 = a * dx * dx + 2.0 * b * dx * dy + c * dy * dy
                        d2[invalid] = np.inf
                        if sel.invert:
                            mask |= d2 <= (sel.sigma * sel.sigma)
                        else:
                            mask |= d2 > (sel.sigma * sel.sigma)
            except Exception as e:
                logging.warning("Selection mask error: %s", e)
        
        # Mask NaN/Inf on axis columns
        for orig_idx in axis_indices:
            new_idx = index_map.get(orig_idx)
            if new_idx is None:
                continue
            col = np.ascontiguousarray(subset[new_idx, :], dtype=np.float64)
            if _HAVE_NUMBA and _mask_nan_inf_numba is not None:
                _mask_nan_inf_numba(col, mask, mask_nan, mask_inf)
            else:
                if mask_nan:
                    mask |= np.isnan(col)
                if mask_inf:
                    mask |= np.isinf(col)
        
        return mask
    
    # ---- Merge helpers ----
    
    def merge(self, other_source: "ArrowDataSource", mode: str = 'columns') -> bool:
        """
        Merge data from another ArrowDataSource.
        
        Parameters
        ----------
        other_source : ArrowDataSource
        mode : {'columns', 'rows'}
        
        Returns
        -------
        bool
            True on success, False otherwise.
        """
        def _warn(title: str, msg: str) -> None:
            try:
                from qtpy.QtWidgets import QMessageBox
                QMessageBox.warning(None, title, msg)
            except Exception:
                print(f"[merge:{title}] {msg}", file=sys.stderr)
        
        if self._table is None:
            self._set_table(other_source._table)
            return True
        
        if other_source._table is None:
            return True
        
        if mode == 'columns':
            if self._table.num_rows != other_source._table.num_rows:
                _warn(
                    "Row Count Mismatch",
                    f"New data has {other_source._table.num_rows} rows, "
                    f"current has {self._table.num_rows} rows."
                )
                return False
            
            # Add non-duplicate columns
            existing_cols = set(self._table.column_names)
            new_cols = []
            for col_name in other_source._table.column_names:
                if col_name not in existing_cols:
                    new_cols.append(other_source._table.column(col_name))
                    existing_cols.add(col_name)
            
            if new_cols:
                # Append columns to table
                for i, col_name in enumerate(other_source._table.column_names):
                    if col_name not in set(self._table.column_names):
                        self._table = self._table.append_column(
                            col_name,
                            other_source._table.column(col_name)
                        )
                self._parameter_names = list(self._table.column_names)
                self._cached_values_array = None
                self._column_arrays.clear()
            return True
        
        if mode == 'rows':
            existing = set(self._table.column_names)
            incoming = set(other_source._table.column_names)
            common = sorted(existing.intersection(incoming))
            
            if not common:
                _warn("No Common Columns", "No overlapping columns. Cannot append rows.")
                return False
            
            # Select common columns and concatenate
            self_common = self._table.select(common)
            other_common = other_source._table.select(common)
            combined = pa.concat_tables([self_common, other_common])
            self._set_table(combined)
            return True
        
        _warn("Invalid Merge Mode", f"Invalid mode: {mode}")
        return False


# Factory function to create the best available data source
def create_data_source(
    parameter_names: Optional[List[str]] = None,
    data: Optional[Any] = None,
    prefer_arrow: bool = True,
) -> Any:
    """
    Create a data source using the best available backend.
    
    Parameters
    ----------
    parameter_names : Optional[List[str]]
        Column names
    data : Optional[Any]
        Initial data (DataFrame, ndarray, etc.)
    prefer_arrow : bool
        If True and PyArrow is available, use ArrowDataSource
    
    Returns
    -------
    DataSource or ArrowDataSource
    """
    if prefer_arrow and HAVE_PYARROW:
        try:
            return ArrowDataSource(parameter_names=parameter_names, data=data)
        except Exception as e:
            logging.warning("Failed to create ArrowDataSource: %s. Falling back to DataSource.", e)
    
    # Fall back to pandas-based DataSource
    from .data_source import DataSource
    return DataSource(parameter_names=parameter_names, data=data)
