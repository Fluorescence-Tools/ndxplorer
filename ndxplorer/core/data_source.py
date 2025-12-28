#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data utilities: case-insensitive column lookup, computed columns from formulas,
and selection masks (rectangular & 2D Gaussian). Includes a DataSource wrapper.

Key improvements
---------------
- De-duplicated imports & added type hints/docstrings.
- Robust equation-file loading (YAML or JSON by extension).
- Constant handling fixed: quoted names that match constants are wrapped as c['Name'].
- Safer evaluation: try pandas.eval (engine='python'), fall back to plain eval.
- Case-insensitive + "left-of-pipe" column matching preserved.
- DataSource cache invalidation and merge helpers retained and clarified.
"""

from __future__ import annotations

import abc
import json
import sys
import re
from typing import Dict, List, Optional, Iterable, Any, Set, Tuple
from collections import OrderedDict

import numpy as np
import pandas as pd

from ..logging_config import logging

try:
    import yaml  # optional
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

# Optional Numba acceleration
try:
    import numba as nb
    _HAVE_NUMBA = True
except ImportError:
    nb = None
    _HAVE_NUMBA = False

# Optional PyArrow for faster numeric conversion
try:
    import pyarrow as pa
    import pyarrow.compute as pc
    _HAVE_PYARROW = True
except ImportError:
    pa = None
    pc = None
    _HAVE_PYARROW = False


# ----------------------------------------
# Numba-accelerated mask computation
# ----------------------------------------

if _HAVE_NUMBA:
    @nb.njit(cache=True, parallel=True, fastmath=True)
    def _rectangular_mask_numba(
        vals: np.ndarray,
        lower: float,
        upper: float,
        invert: bool,
        mask: np.ndarray,
    ) -> None:
        """Apply rectangular selection mask in-place using Numba."""
        n = vals.shape[0]
        for i in nb.prange(n):
            v = vals[i]
            if invert:
                if v > lower and v < upper:
                    mask[i] = True
            else:
                if v < lower or v > upper:
                    mask[i] = True

    @nb.njit(cache=True, fastmath=True)
    def _gaussian2d_mask_numba(
        x: np.ndarray,
        y: np.ndarray,
        mu0: float,
        mu1: float,
        inv_cov00: float,
        inv_cov01: float,
        inv_cov11: float,
        sigma_sq: float,
        invert: bool,
        log_x: bool,
        log_y: bool,
        mask: np.ndarray,
    ) -> None:
        """Apply Gaussian 2D selection mask in-place using Numba."""
        n = x.shape[0]
        for i in range(n):
            xv = x[i]
            yv = y[i]
            
            # Apply log transform if needed
            if log_x:
                if xv > 0.0:
                    xv = np.log(xv)
                else:
                    mask[i] = True
                    continue
            if log_y:
                if yv > 0.0:
                    yv = np.log(yv)
                else:
                    mask[i] = True
                    continue
            
            # Check for invalid values
            if not np.isfinite(xv) or not np.isfinite(yv):
                mask[i] = True
                continue
            
            dx = xv - mu0
            dy = yv - mu1
            d2 = inv_cov00 * dx * dx + 2.0 * inv_cov01 * dx * dy + inv_cov11 * dy * dy
            
            if invert:
                if d2 <= sigma_sq:
                    mask[i] = True
            else:
                if d2 > sigma_sq:
                    mask[i] = True

    @nb.njit(cache=True, parallel=True, fastmath=True)
    def _mask_nan_inf_numba(col: np.ndarray, mask: np.ndarray, do_nan: bool, do_inf: bool) -> None:
        """Mask NaN and/or Inf values in-place."""
        n = col.shape[0]
        for i in nb.prange(n):
            v = col[i]
            if do_nan and np.isnan(v):
                mask[i] = True
            elif do_inf and np.isinf(v):
                mask[i] = True

else:
    _rectangular_mask_numba = None
    _gaussian2d_mask_numba = None
    _mask_nan_inf_numba = None


# ---------------------------
# Fast numeric conversion
# ---------------------------

def _fast_to_numeric(df: pd.DataFrame, use_float32: bool = True) -> pd.DataFrame:
    """
    Convert DataFrame columns to numeric efficiently.
    
    Uses PyArrow when available for ~2-5x faster conversion on large DataFrames.
    Falls back to pandas apply() otherwise.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with potentially mixed types
    use_float32 : bool
        If True (default), use float32 to halve memory usage.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with all columns converted to numeric (non-numeric → NaN)
    """
    if df.empty:
        return df.copy()
    
    import time
    t0 = time.perf_counter()
    
    # Target dtype for memory efficiency
    target_dtype = np.float32 if use_float32 else np.float64
    pa_target = pa.float32() if use_float32 else pa.float64()
    
    if _HAVE_PYARROW:
        try:
            # Convert to Arrow Table for fast processing
            table = pa.Table.from_pandas(df, preserve_index=False)
            
            # Convert each column to target float type
            new_columns = []
            for i, col_name in enumerate(table.column_names):
                col = table.column(i)
                col_type = col.type
                
                # If already numeric, cast to target type
                if pa.types.is_floating(col_type) or pa.types.is_integer(col_type):
                    new_columns.append(pc.cast(col, pa_target, safe=False))
                elif pa.types.is_boolean(col_type):
                    new_columns.append(pc.cast(col, pa_target, safe=False))
                else:
                    # String or other type: try to convert
                    try:
                        # Use Arrow's string-to-float conversion
                        new_columns.append(pc.cast(col, pa_target, safe=False))
                    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                        # Fall back to pandas for this column
                        series = col.to_pandas()
                        numeric_series = pd.to_numeric(series, errors='coerce').astype(target_dtype)
                        new_columns.append(pa.array(numeric_series.values))
            
            # Reconstruct table and convert back to pandas
            result_table = pa.Table.from_arrays(new_columns, names=table.column_names)
            result = result_table.to_pandas(
                self_destruct=True,
                split_blocks=True,
                zero_copy_only=False,
            )
            
            t1 = time.perf_counter()
            logging.debug("[_fast_to_numeric] PyArrow (%s): %d rows × %d cols in %.3fs",
                         'float32' if use_float32 else 'float64',
                         len(df), len(df.columns), t1 - t0)
            return result
            
        except Exception as e:
            logging.debug("[_fast_to_numeric] PyArrow failed: %s, falling back to pandas", e)
    
    # Fallback to pandas (still optimized)
    result = df.copy()
    for col in result.columns:
        if not pd.api.types.is_numeric_dtype(result[col]):
            result[col] = pd.to_numeric(result[col], errors='coerce').astype(target_dtype)
        elif result[col].dtype != target_dtype:
            result[col] = result[col].astype(target_dtype)
    
    t1 = time.perf_counter()
    logging.debug("[_fast_to_numeric] pandas (%s): %d rows × %d cols in %.3fs",
                 'float32' if use_float32 else 'float64',
                 len(df), len(df.columns), t1 - t0)
    return result


# ---------------------------
# Case-insensitive DataFrame accessor
# ---------------------------

class CaseInsensitiveDict:
    """
    A very small wrapper that allows case-insensitive access to a pandas.DataFrame
    via indexing (d['ColName']). For columns with suffixes like 'Name | 0-2048',
    the matcher also compares the left part before the first '|'.
    
    Optimized for large datasets with cached column lookups.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self._column_cache = {}
        self._cache_valid = False
        self._build_cache()

    def _build_cache(self):
        """Build optimized lookup cache for column access."""
        self._column_cache = {}
        self._column_cache['exact'] = {}
        self._column_cache['left_pipe'] = {}
        self._column_cache['prefix'] = {}
        
        for col in self.data.columns:
            col_str = str(col)
            col_lower = col_str.lower()
            left = col_str.split('|', 1)[0].strip().lower()
            
            # Store exact matches
            self._column_cache['exact'][col_lower] = col
            
            # Store left-of-pipe matches
            if left not in self._column_cache['left_pipe']:
                self._column_cache['left_pipe'][left] = col
            
            # Store prefix matches (for fallback)
            for prefix_len in range(1, min(10, len(col_lower)) + 1):
                prefix = col_lower[:prefix_len]
                if prefix not in self._column_cache['prefix']:
                    self._column_cache['prefix'][prefix] = col
        
        self._cache_valid = True

    def __getitem__(self, key: Any):
        if isinstance(key, str) and isinstance(self.data, pd.DataFrame):
            if not self._cache_valid:
                self._build_cache()
                
            k_lower = key.lower().strip()
            
            # Try exact match first
            if k_lower in self._column_cache['exact']:
                col = self._column_cache['exact'][k_lower]
                val = self.data[col]
                return pd.to_numeric(val, errors='coerce') if not pd.api.types.is_numeric_dtype(val) else val
            
            # Try left-of-pipe match
            if k_lower in self._column_cache['left_pipe']:
                col = self._column_cache['left_pipe'][k_lower]
                val = self.data[col]
                return pd.to_numeric(val, errors='coerce') if not pd.api.types.is_numeric_dtype(val) else val
            
            # Fallback: prefix match
            if k_lower in self._column_cache['prefix']:
                col = self._column_cache['prefix'][k_lower]
                val = self.data[col]
                return pd.to_numeric(val, errors='coerce') if not pd.api.types.is_numeric_dtype(val) else val
            
            # Let pandas raise if nothing matched
            return self.data[key]
        return self.data[key]


# ---------------------------
# Equation application
# ---------------------------

def _load_equations_file(path: str) -> List[Dict[str, str]]:
    """
    Load equations from a YAML or JSON file. The file is expected to contain a list
    of mappings like: [{"Fg": "'Sg' - 'Bg'"}, {"Proximity ratio": "'Sr' / ('Sg' + 'Sr')"}]
    """
    with open(path, "r", encoding="utf-8") as fp:
        text = fp.read()

    # Decide by extension first, fallback to a safe YAML if available, else JSON
    lower = path.lower()
    if lower.endswith(".json"):
        return json.loads(text, object_pairs_hook=OrderedDict)
    if yaml is not None:
        return yaml.safe_load(text)  # type: ignore
    # As a last resort, try JSON
    return json.loads(text, object_pairs_hook=OrderedDict)


def compute_values(
    d: pd.DataFrame,
    constants: Dict[str, float],
    equations: Optional[List[Dict[str, str]]] = None,
    equation_json_fn: Optional[str] = None,
    engine: str = "python",
) -> None:
    """
    Compute columns in DataFrame `d` from `equations`, using case-insensitive
    column lookup and quoted-name replacement for data/constant references.

    Parameters
    ----------
    d : pd.DataFrame
        The table to augment; new columns are added/overwritten in-place.
    constants : Dict[str, float]
        Name → value constants. Accessed in formulas as c['Name'].
    equations : Optional[List[Dict[str, str]]]
        List of {new_column_name: "expression"} dicts. If None, taken from file.
    equation_json_fn : Optional[str]
        Path to YAML/JSON file with equations (detected by extension).
    engine : str
        Passed to pandas.eval. Use 'python' (default) for widest syntax support.

    Notes
    -----
    - Expressions may refer to columns or constants using *quoted* names:
        'Sg' / 'Sr'               -> columns
        'Bg'                      -> constant (if present in `constants`)
    - The preprocessor will auto-wrap quoted names not already written
      as d['...'] or c['...'] into the appropriate form (favoring data columns).
    """
    c = constants

    if equations is None and equation_json_fn:
        try:
            equations = _load_equations_file(equation_json_fn)
        except Exception as e:
            logging.warning(f"compute_values: Failed to load equations from {equation_json_fn}: {e}")
            equations = []

    equations = equations or []

    # Collect lowercase keys defined by equations to enable forward references
    eq_keys_lower = {str(k).lower() for m in equations for k in m.keys()}

    # Precompute lookups
    def _normalize_left(s: str) -> str:
        return str(s).split('|', 1)[0].strip()

    cols_lower_exact = {str(col).lower() for col in d.columns}
    cols_lower_left = {_normalize_left(col).lower() for col in d.columns}
    consts_lower = {str(name).lower() for name in c.keys()}

    def _preprocess_equation(eq_str: str) -> Tuple[str, Set[str], Set[str]]:
        """
        Replace occurrences of 'name' / "name" with d['name'] or c['name'] depending on
        whether it's a column/equation key or constant, unless already inside d[...] or c[...].
        """
        if not isinstance(eq_str, str) or not eq_str:
            return eq_str, set(), set()

        out, i = [], 0
        data_refs: Set[str] = set()
        const_refs: Set[str] = set()
        for m in re.finditer(r"(['\"])\s*(.*?)\s*\1", eq_str):
            s, e = m.span()
            name = m.group(2)
            out.append(eq_str[i:s])

            # Check if already wrapped as d['...'] or c['...']
            j = s - 1
            while j >= 0 and eq_str[j].isspace():
                j -= 1
            is_wrapped = False
            if j >= 0 and eq_str[j] == '[':
                k = j - 1
                while k >= 0 and eq_str[k].isspace():
                    k -= 1
                if k >= 0 and eq_str[k] in ('d', 'c'):
                    is_wrapped = True

            if is_wrapped:
                out.append(eq_str[s:e])
                ref_type = eq_str[k]
                lname = name.lower()
                if ref_type == 'd':
                    data_refs.add(lname)
                elif ref_type == 'c':
                    const_refs.add(lname)
            else:
                lname = name.lower()
                lname_left = _normalize_left(name).lower()
                if (lname in cols_lower_exact) or (lname_left in cols_lower_left) or (lname in eq_keys_lower):
                    out.append(f"d['{name}']")
                    data_refs.add(lname)
                elif lname in consts_lower:
                    out.append(f"c['{name}']")
                    const_refs.add(lname)
                else:
                    # Treat unknown quoted names as data references to force failure if missing
                    out.append(f"d['{name}']")
                    data_refs.add(lname)

            i = e

        out.append(eq_str[i:])
        return ''.join(out), data_refs, const_refs

    d_ci = CaseInsensitiveDict(d)
    available_cols_exact = set(cols_lower_exact)
    available_cols_left = set(cols_lower_left)

    for mapping in equations:
        for out_key, expr in mapping.items():
            try:
                pre, data_refs, const_refs = _preprocess_equation(expr)
                missing_cols = []
                for ref in data_refs:
                    if (
                        ref not in available_cols_exact
                        and _normalize_left(ref).lower() not in available_cols_left
                        and ref not in eq_keys_lower
                    ):
                        missing_cols.append(ref)
                missing_consts = [ref for ref in const_refs if ref not in consts_lower]
                if missing_cols or missing_consts:
                    missing_desc = []
                    if missing_cols:
                        missing_desc.append(f"columns={sorted(set(missing_cols))}")
                    if missing_consts:
                        missing_desc.append(f"constants={sorted(set(missing_consts))}")
                    logging.info(
                        "compute_values: Skipping '%s' due to missing %s",
                        out_key,
                        "; ".join(missing_desc),
                    )
                    continue
                # First try pandas.eval (engine='python' supports general Python eval)
                try:
                    d[out_key] = pd.eval(pre, local_dict={'d': d_ci, 'c': c}, engine=engine)
                except Exception:
                    # Fallback to plain eval for maximum compatibility
                    d[out_key] = eval(pre, {}, {'d': d_ci, 'c': c})
                lower_key = str(out_key).lower()
                available_cols_exact.add(lower_key)
                available_cols_left.add(_normalize_left(out_key).lower())
            except Exception as e:
                logging.warning(f"compute_values: Could not compute '{out_key}': {e}")


# ---------------------------
# Selection API
# ---------------------------

class DataSelection(abc.ABC):
    @abc.abstractmethod
    def get_mask(self, data: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        data : np.ndarray, shape (n_parameters, n_points)

        Returns
        -------
        mask : np.ndarray (bool), same shape as data
            True means "masked out" (excluded).
        """
        raise NotImplementedError


class Gaussian2DSelection(DataSelection):
    """
    Elliptical selection in 2D using Mahalanobis distance around mean `mu`
    with covariance `cov`. Supports optional per-axis log transforms.

    If invert=False (default): mask points OUTSIDE the ellipse (d2 > sigma^2).
    If invert=True:  mask points INSIDE the ellipse (d2 <= sigma^2).
    """

    def __init__(
        self,
        parameter_idx1: int,
        parameter_idx2: int,
        mu: Iterable[float],
        cov: Iterable[Iterable[float]],
        sigma: float = 1.0,
        invert: bool = False,
        enabled: bool = True,
        name: Optional[str] = None,
        log_x: bool = False,
        log_y: bool = False,
    ):
        self.parameter_idx1 = int(parameter_idx1)
        self.parameter_idx2 = int(parameter_idx2)
        self.mu = np.asarray(mu, dtype=float).reshape(2)
        self.cov = np.asarray(cov, dtype=float).reshape(2, 2)
        self.sigma = float(sigma)
        self.invert = bool(invert)
        self.enabled = bool(enabled)
        self.name = name
        self.log_x = bool(log_x)
        self.log_y = bool(log_y)

    def get_mask(self, data: np.ndarray) -> np.ndarray:
        n_param, n_pts = data.shape
        mask = np.zeros((n_param, n_pts), dtype=bool)
        if not self.enabled:
            return mask
        if self.parameter_idx1 >= n_param or self.parameter_idx2 >= n_param:
            return mask

        x = data[self.parameter_idx1, :]
        y = data[self.parameter_idx2, :]

        with np.errstate(divide='ignore', invalid='ignore'):
            zx = np.where(x > 0.0, np.log(x), np.nan) if self.log_x else x.astype(float)
            zy = np.where(y > 0.0, np.log(y), np.nan) if self.log_y else y.astype(float)

        try:
            inv_cov = np.linalg.inv(self.cov)
        except Exception:
            inv_cov = np.linalg.pinv(self.cov)

        dx = zx - self.mu[0]
        dy = zy - self.mu[1]
        invalid = ~np.isfinite(dx) | ~np.isfinite(dy)
        dx = np.nan_to_num(dx, nan=np.inf)
        dy = np.nan_to_num(dy, nan=np.inf)

        a = inv_cov[0, 0]
        b = inv_cov[0, 1]
        c = inv_cov[1, 1]
        d2 = a * dx * dx + 2.0 * b * dx * dy + c * dy * dy
        d2[invalid] = np.inf

        if self.invert:
            out_of_bounds = d2 <= (self.sigma * self.sigma)
        else:
            out_of_bounds = d2 > (self.sigma * self.sigma)

        mask[:, out_of_bounds] = True
        return mask


class RectangularDataSelection(DataSelection):
    """
    Simple 1D interval selection on a chosen parameter index.

    If invert=False (default): mask values outside [lower, upper].
    If invert=True:  mask values inside (lower, upper) (open interval).
    """

    def __init__(
        self,
        parameter_idx: int,
        lower: float,
        upper: float,
        invert: bool = False,
        enabled: bool = True,
        name: Optional[str] = None,
    ):
        self.parameter_idx = int(parameter_idx)
        self.lower = float(lower)
        self.upper = float(upper)
        self.invert = bool(invert)
        self.enabled = bool(enabled)
        self.name = name

    def __str__(self) -> str:  # pragma: no cover
        return (f"RectangularDataSelection:\nBounds: {self.lower}, {self.upper}\n"
                f"Invert: {self.invert}\nEnabled: {self.enabled}\n")

    def get_mask(self, data: np.ndarray) -> np.ndarray:
        n_param, n_pts = data.shape
        mask = np.zeros((n_param, n_pts), dtype=bool)
        if not self.enabled:
            return mask
        if self.parameter_idx >= n_param:
            print(f"Parameter idx {self.parameter_idx} exceeds dimension {n_param}.", file=sys.stderr)
            return mask

        vals = data[self.parameter_idx, :]
        if self.invert:
            bad = (vals > self.lower) & (vals < self.upper)  # mask inside
        else:
            bad = (vals < self.lower) | (vals > self.upper)  # mask outside
        mask[:, bad] = True
        return mask


# ---------------------------
# DataSource wrapper
# ---------------------------

class DataSource:
    """
    Light wrapper around a DataFrame that provides:
    - cached numeric values (transposed) for fast selection operations,
    - computed columns from equations/constants,
    - merge (by columns or rows) convenience,
    - masking utilities that combine multiple selections and NaN/Inf culling.
    - **column filtering** for operating on only relevant columns (axes + selections)
    """

    _data: pd.DataFrame
    _data_numeric: pd.DataFrame
    _parameter_names: List[str]
    _relevant_columns_cache: Optional[Tuple[Tuple[int, ...], np.ndarray]] = None

    def __init__(self, parameter_names: Optional[List[str]] = None, data: Optional[pd.DataFrame | np.ndarray] = None):
        # Performance optimization: initialize cache before data assignment
        self._column_cache = {}
        self._cache_valid = False
        self._cached_values_array = None
        
        if isinstance(data, np.ndarray):
            self.data = pd.DataFrame(data, columns=parameter_names)
        elif isinstance(data, pd.DataFrame):
            self.data = data
        else:
            self.data = pd.DataFrame()

        if isinstance(parameter_names, list):
            self._parameter_names = parameter_names
        else:
            self._parameter_names = list(self._data.columns)

    def __str__(self) -> str:  # pragma: no cover
        return self._data.__str__()

    def __len__(self) -> int:
        return self.size

    # ---- properties ----

    @property
    def parameter_names(self) -> List[str]:
        return self._parameter_names

    @property
    def values(self) -> np.ndarray:
        """
        Returns (n_parameters, n_points) numeric np.ndarray (cached).
        Optimized for large datasets with lazy evaluation and memory efficiency.
        Uses float32 to halve memory usage compared to float64.
        
        The transposed array is cached to avoid repeated memory copies.
        """
        if self._cached_values_array is not None:
            return self._cached_values_array
        
        # Get underlying numpy array - avoid DataFrame overhead
        numeric_data = self._data_numeric.values
        
        # Convert to float32 only if needed (halves memory vs float64)
        if numeric_data.dtype != np.float32:
            # Use Fortran order for the transposed result to be C-contiguous
            self._cached_values_array = np.ascontiguousarray(
                numeric_data.T, dtype=np.float32
            )
        else:
            # If already float32, just transpose with contiguous memory
            self._cached_values_array = np.ascontiguousarray(numeric_data.T)
        
        return self._cached_values_array

    def clear(self) -> None:
        self.data = pd.DataFrame()

    def compute_columns(
        self,
        constants: Dict[str, float],
        equations: Optional[List[Dict[str, str]]] = None,
        equation_json_fn: Optional[str] = None,
        engine: str = "python",
    ) -> None:
        compute_values(
            d=self.data,
            constants=constants,
            equations=equations,
            equation_json_fn=equation_json_fn,
            engine=engine,
        )
        # Ensure caches are refreshed
        self.data = self.data

    @property
    def empty(self) -> bool:
        return self._data.empty

    def get_mask(
        self,
        selections: List[DataSelection],
        idxs: Optional[List[int]] = None,
        mask_nan: bool = True,
        mask_inf: bool = True,
    ) -> np.ndarray:
        """
        Combine selection masks and optionally mask NaN/Inf on selected parameter indices.
        Optimized for large datasets with vectorized operations and early termination.

        Returns
        -------
        mask : np.ndarray (bool), shape (n_parameters, n_points)
            True → masked/excluded.
        """
        idxs = idxs or []
        d = self.values
        n_param, n_pts = d.shape
        
        # Pre-allocate mask with zeros for better performance
        mask = np.zeros((n_param, n_pts), dtype=bool)
        
        # Early exit if no selections and no idx filtering
        if not selections and not idxs:
            return mask
        
        # Process selections with vectorized operations
        for sel in selections:
            try:
                m = sel.get_mask(d)
                if isinstance(m, np.ndarray) and m.shape == mask.shape:
                    # Use in-place OR operation for better performance
                    mask |= m
            except Exception as e:
                print(f"[DataSource.get_mask] Selection error ({getattr(sel, 'name', 'unnamed')}): {e}", file=sys.stderr)

        # Vectorized NaN/Inf filtering for selected indices
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

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @data.setter
    def data(self, v: pd.DataFrame) -> None:
        # Avoid copy if v is already a DataFrame and caller doesn't need original
        # For large datasets, this saves significant memory and time
        if isinstance(v, pd.DataFrame):
            # Always copy to ensure we own the data and avoid unexpected mutations
            # The copy is necessary for correctness but we optimize the numeric conversion
            self._data = v.copy()
        else:
            self._data = pd.DataFrame()
        self._parameter_names = list(self._data.columns)
        # Optimized numeric conversion using PyArrow when available
        self._data_numeric = _fast_to_numeric(self._data)
        # Invalidate all caches
        self._cached_values_array = None
        self._cache_valid = False
        if hasattr(self, '_column_cache'):
            self._column_cache.clear()
        # Invalidate column subset cache
        self._relevant_columns_cache = None

    @property
    def size(self) -> int:
        return self.values.shape[1] if not self.empty else 0

    # ---- column filtering for performance ----

    def get_relevant_column_indices(
        self,
        axis_indices: List[int],
        selections: List[DataSelection],
        extra_indices: Optional[List[int]] = None,
    ) -> List[int]:
        """
        Collect all column indices that are actually needed for current operations.

        Parameters
        ----------
        axis_indices : List[int]
            Indices of axis columns (x, y, z, weight, etc.).
        selections : List[DataSelection]
            Current selections that reference columns by index.
        extra_indices : Optional[List[int]]
            Any additional column indices to include.

        Returns
        -------
        List[int]
            Sorted, unique list of column indices needed.
        """
        indices: Set[int] = set(axis_indices)
        if extra_indices:
            indices.update(extra_indices)

        for sel in selections:
            if isinstance(sel, RectangularDataSelection):
                indices.add(sel.parameter_idx)
            elif isinstance(sel, Gaussian2DSelection):
                indices.add(sel.parameter_idx1)
                indices.add(sel.parameter_idx2)

        n_cols = len(self._parameter_names)
        return sorted(idx for idx in indices if 0 <= idx < n_cols)

    def get_values_subset(
        self,
        column_indices: List[int],
    ) -> Tuple[np.ndarray, Dict[int, int]]:
        """
        Return a subset of the values array containing only the specified columns.

        Parameters
        ----------
        column_indices : List[int]
            Original column indices to include.

        Returns
        -------
        subset : np.ndarray
            Shape (len(column_indices), n_points) with only the requested columns.
        index_map : Dict[int, int]
            Mapping from original column index to new index in the subset.
        """
        cache_key = tuple(column_indices)
        if (
            hasattr(self, '_relevant_columns_cache')
            and self._relevant_columns_cache is not None
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
        selections: List[DataSelection],
        axis_indices: List[int],
        mask_nan: bool = True,
        mask_inf: bool = True,
    ) -> np.ndarray:
        """
        Compute mask using only the relevant columns for better performance.

        This is an optimized version of get_mask that first filters to only
        the columns referenced by selections and axes, reducing memory and
        computation for large datasets with many columns.
        
        Uses Numba JIT compilation when available for ~10x speedup.

        Parameters
        ----------
        selections : List[DataSelection]
            Current selections.
        axis_indices : List[int]
            Indices of axis columns (x, y, z).
        mask_nan : bool
            Whether to mask NaN values.
        mask_inf : bool
            Whether to mask Inf values.

        Returns
        -------
        mask : np.ndarray (bool), shape (n_points,)
            1D mask where True means the point should be excluded.
        """
        relevant_indices = self.get_relevant_column_indices(axis_indices, selections)
        if not relevant_indices:
            return np.zeros(self.size, dtype=bool)

        subset, index_map = self.get_values_subset(relevant_indices)
        n_pts = subset.shape[1]
        mask = np.zeros(n_pts, dtype=bool)

        for sel in selections:
            if not getattr(sel, 'enabled', True):
                continue
            try:
                if isinstance(sel, RectangularDataSelection):
                    new_idx = index_map.get(sel.parameter_idx)
                    if new_idx is None:
                        continue
                    vals = np.ascontiguousarray(subset[new_idx, :], dtype=np.float64)
                    
                    # Use Numba if available
                    if _HAVE_NUMBA and _rectangular_mask_numba is not None:
                        _rectangular_mask_numba(vals, sel.lower, sel.upper, sel.invert, mask)
                    else:
                        if sel.invert:
                            mask |= (vals > sel.lower) & (vals < sel.upper)
                        else:
                            mask |= (vals < sel.lower) | (vals > sel.upper)
                            
                elif isinstance(sel, Gaussian2DSelection):
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
                    
                    # Use Numba if available
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
                logging.warning("Selection mask error (%s): %s", getattr(sel, 'name', 'unnamed'), e)

        # Mask NaN/Inf on axis columns - use Numba if available
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

    # ---- merge helpers ----

    def merge(self, other_source: "DataSource", mode: str = 'columns') -> bool:
        """
        Merge data from another DataSource.

        Parameters
        ----------
        other_source : DataSource
        mode : {'columns', 'rows'}

        Returns
        -------
        bool
            True on success, False otherwise.
        """
        # Lazy import to avoid hard dependency on Qt in headless environments
        def _warn(title: str, msg: str) -> None:
            try:
                from qtpy.QtWidgets import QMessageBox  # type: ignore
                QMessageBox.warning(None, title, msg)
            except Exception:
                print(f"[merge:{title}] {msg}", file=sys.stderr)

        if mode == 'columns':
            if len(self.data) != len(other_source.data):
                _warn(
                    "Row Count Mismatch",
                    f"New data has {len(other_source.data)} rows, current has {len(self.data)} rows. Not merging."
                )
                return False

            duplicate_cols = set(self.data.columns).intersection(set(other_source.data.columns))
            df_unique = other_source.data.drop(columns=list(duplicate_cols)) if duplicate_cols else other_source.data
            combined = pd.concat([self.data, df_unique], axis=1)
            self.data = combined
            return True

        if mode == 'rows':
            existing = set(self.data.columns)
            incoming = set(other_source.data.columns)
            new_unique = incoming - existing
            if new_unique:
                _warn(
                    "New Columns Found",
                    f"New data contains columns not in current data: {', '.join(sorted(new_unique))}. "
                    f"Only rows of existing columns will be appended."
                )

            common = sorted(existing.intersection(incoming))
            if not common:
                _warn("No Common Columns", "No overlapping columns. Cannot append rows.")
                return False

            other_common = other_source.data[common]
            combined = pd.concat([self.data[common], other_common], axis=0, ignore_index=True)
            # Keep original full set of columns if desired; here we keep only common to ensure consistency
            self.data = combined
            return True

        _warn("Invalid Merge Mode", f"Invalid mode: {mode}. Must be 'columns' or 'rows'.")
        return False
