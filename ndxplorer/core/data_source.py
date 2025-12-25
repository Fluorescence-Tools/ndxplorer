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
from typing import Dict, List, Optional, Iterable, Any
from collections import OrderedDict

import numpy as np
import pandas as pd

from ..logging_config import logging

try:
    import yaml  # optional
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


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

    def _preprocess_equation(eq_str: str) -> str:
        """
        Replace occurrences of 'name' / "name" with d['name'] or c['name'] depending on
        whether it's a column/equation key or constant, unless already inside d[...] or c[...].
        """
        if not isinstance(eq_str, str) or not eq_str:
            return eq_str

        out, i = [], 0
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
            else:
                lname = name.lower()
                lname_left = _normalize_left(name).lower()
                if (lname in cols_lower_exact) or (lname_left in cols_lower_left) or (lname in eq_keys_lower):
                    out.append(f"d['{name}']")
                elif lname in consts_lower:
                    out.append(f"c['{name}']")
                else:
                    # Treat unknown quoted names as data references to force failure if missing
                    out.append(f"d['{name}']")

            i = e

        out.append(eq_str[i:])
        return ''.join(out)

    d_ci = CaseInsensitiveDict(d)

    for mapping in equations:
        for out_key, expr in mapping.items():
            try:
                pre = _preprocess_equation(expr)
                # First try pandas.eval (engine='python' supports general Python eval)
                try:
                    d[out_key] = pd.eval(pre, local_dict={'d': d_ci, 'c': c}, engine=engine)
                except Exception:
                    # Fallback to plain eval for maximum compatibility
                    d[out_key] = eval(pre, {}, {'d': d_ci, 'c': c})
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
    """

    _data: pd.DataFrame
    _data_numeric: pd.DataFrame
    _parameter_names: List[str]

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
        """
        if not hasattr(self, '_cached_values_array') or self._cached_values_array is None:
            # Use memory-efficient conversion without intermediate copy
            numeric_data = self._data_numeric.values
            if numeric_data.dtype != np.float64:
                numeric_data = numeric_data.astype(np.float64, copy=False)
            # Transpose in-place when possible
            self._cached_values_array = numeric_data.T
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
            valid_idxs = [idx for idx in idxs if 0 <= idx < n_param]
            if valid_idxs:
                # Process all valid indices at once for better performance
                for idx in valid_idxs:
                    col = d[idx, :]
                    bad_mask = np.zeros(n_pts, dtype=bool)
                    if mask_nan:
                        bad_mask |= np.isnan(col)
                    if mask_inf:
                        bad_mask |= np.isinf(col)
                    # Apply column-wise mask to all parameters
                    mask[:, bad_mask] = True

        return mask

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @data.setter
    def data(self, v: pd.DataFrame) -> None:
        self._data = v.copy() if isinstance(v, pd.DataFrame) else pd.DataFrame()
        self._parameter_names = list(self._data.columns)
        self._data_numeric = self._data.apply(pd.to_numeric, errors='coerce')
        # Invalidate all caches
        if hasattr(self, '_cached_values_array'):
            self._cached_values_array = None
        self._cache_valid = False
        if hasattr(self, '_column_cache'):
            self._column_cache.clear()

    @property
    def size(self) -> int:
        return self.values.shape[1] if not self.empty else 0

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
