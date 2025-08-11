from __future__ import print_function
from typing import List, Dict

import sys
import yaml
import abc
import numpy as np
import pandas as pd
import re

from collections import OrderedDict


class CaseInsensitiveDict:
    """A dictionary wrapper that allows case-insensitive access to keys."""

    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        if isinstance(key, str) and isinstance(self.data, pd.DataFrame):
            # For DataFrame, try to find a case-insensitive match among column names
            for col in self.data.columns:
                if col.lower() == key.lower():
                    return self.data[col]
            
            # If exact case-insensitive match not found, try pattern matching
            # Look for columns that contain the key as a prefix (before any separator like '|')
            if isinstance(key, str):
                for col in self.data.columns:
                    # First, check for equality of the left part (before any '|') to be precise
                    parts = col.lower().split('|')
                    if parts and parts[0].strip() == key.lower().strip():
                        return self.data[col]
                    # Then fallback to startswith matching (case insensitive)
                    if col.lower().startswith(key.lower()):
                        return self.data[col]
            
            # If no match found, fall back to original key
            return self.data[key]
        return self.data[key]


def compute_values(
        d,  # type: pd.DataFrame
        constants,  # type: Dict[str, float]
        equations,  # type: List[Dict[str, str]]
        equation_json_fn=None,  # type: str
        engine='python'
):
    # type: (pd.DataFrame, Dict[str, float], List[Dict[str, str]], str, str) -> None
    c = constants
    if equation_json_fn is not None:
        with open(equation_json_fn, "r") as fp:
            equations = yaml.loads(fp.read(), object_pairs_hook=OrderedDict)

    # Build a set of all equation-defined keys (case-insensitive) for generous matching
    eq_keys_lower = set()
    try:
        for _eq in equations or []:
            for _k in _eq.keys():
                eq_keys_lower.add(str(_k).lower())
    except Exception:
        pass

    # Helper: preprocess an equation string to allow direct quoted names
    def _preprocess_equation(eq_str: str) -> str:
        if not isinstance(eq_str, str) or not eq_str:
            return eq_str

        # Normalization helper: take the part before '|' and strip surrounding whitespace
        def _normalize_name(s: str) -> str:
            try:
                left = str(s).split('|', 1)[0]
                return left.strip()
            except Exception:
                return str(s).strip()

        # Build case-insensitive lookup sets
        cols_lower_exact = {str(col).lower() for col in d.columns}
        cols_lower_normalized = {_normalize_name(col).lower() for col in d.columns}
        consts_lower = {str(name).lower() for name in c.keys()}

        # Replace occurrences of 'name' or "name" that are NOT already inside d[...] or c[...]
        out = []
        i = 0
        for m in re.finditer(r"(['\"])\s*(.*?)\s*\1", eq_str):
            s, e = m.span()
            name = m.group(2)
            # Append text before the match
            out.append(eq_str[i:s])

            # Determine if this quoted token is already within d[...] or c[...]
            j = s - 1
            # Skip whitespace backwards
            while j >= 0 and eq_str[j].isspace():
                j -= 1
            is_wrapped = False
            if j >= 0 and eq_str[j] == '[':
                # Skip whitespace before '[' to find the preceding char
                k = j - 1
                while k >= 0 and eq_str[k].isspace():
                    k -= 1
                if k >= 0 and eq_str[k] in ('d', 'c'):
                    is_wrapped = True

            if is_wrapped:
                # Leave as-is
                out.append(eq_str[s:e])
            else:
                # Decide whether it's a data column or a constant (prefer columns)
                lname = str(name).lower()
                lname_norm = _normalize_name(name).lower()
                if (lname in cols_lower_exact) or (lname_norm in cols_lower_normalized) or (lname in eq_keys_lower):
                    # Wrap as data column. We keep the original 'name' as written in the equation;
                    # CaseInsensitiveDict used during eval will resolve it to the real column
                    # (including variants like "... | 0-2048"). If it's an equation-defined key,
                    # the column will exist by the time it is used in subsequent equations.
                    out.append(f"d['{name}']")
                elif lname in consts_lower:
                    out.append(f"c['{name}']")
                else:
                    # Not known: leave literal as-is to avoid breaking strings intentionally used by user
                    out.append(eq_str[s:e])

            i = e
        # Append the tail
        out.append(eq_str[i:])
        return ''.join(out)

    # Wrap the DataFrame with case-insensitive access
    d_case_insensitive = CaseInsensitiveDict(d)

    for eq in equations:
        for key in eq:
            try:
                expr = eq[key]
                expr = _preprocess_equation(expr)
                # Use the original DataFrame for assignment but the wrapper for evaluation
                d[key] = pd.eval(expr, local_dict={'d': d_case_insensitive, 'c': c}, engine=engine)
            except Exception:
                pass
                # print("Could not compute: %s" % key, file=sys.stderr)


class DataSelection(object):

    @abc.abstractmethod
    def get_mask(
            self,
            data  # type: np.ndarray
    ):
        # type: (np.ndarray) -> np.ndarray
        pass


class RectangularDataSelection(DataSelection):

    def __init__(
            self,
            parameter_idx,   # type: int
            lower,  # type: float
            upper,   # type: float
            invert=False,   # type: bool
            enabled=True,   # type: bool
            name=None  # type: str
    ):
        self.parameter_idx = parameter_idx
        self.lower = lower
        self.upper = upper
        self.invert = invert
        self.enabled = enabled
        self.name = name

    def __str__(self):
        s = "RectangularDataSelection:\n"
        s += "Bounds: %s, %s\n" % (self.lower, self.upper)
        s += "Invert: %s\n" % self.invert
        s += "Enabled: %s\n" % self.enabled
        return s

    def get_mask(
            self,
            data  # type: np.ndarray
    ):
        # type: (np.ndarray) -> np.ndarray
        n_parameter, n_data_points = data.shape
        mask = np.ma.make_mask_none((n_parameter, n_data_points))
        s = self
        if s.parameter_idx < n_parameter:
            if s.enabled:
                if s.invert:
                    mask[:, :] |= np.logical_and(data[s.parameter_idx, :] > s.lower, data[s.parameter_idx, :] < s.upper)
                else:
                    mask[:, :] |= np.logical_or(data[s.parameter_idx, :] < s.lower, data[s.parameter_idx, :] > s.upper)
        else:
            print("Parameter with idx %s exceeds dimension of dataset." % s.parameter_idx, file=sys.stderr)
        return mask


class DataSource(object):

    _parameter_names = list()  # type: List[str]
    _data = None  # type: pd.DataFrame
    _data_numeric = None  # type: pd.DataFrame

    def __init__(
            self,
            parameter_names=None,  # type: List[str]
            data=None  # type: np.ndarray
    ):
        # Data
        if isinstance(data, np.ndarray):
            self.data = pd.DataFrame(data, columns=parameter_names)
        elif isinstance(data, pd.DataFrame):
            self.data = data
        else:
            self.data = pd.DataFrame()
        # Parameter names
        if isinstance(parameter_names, list):
            self._parameter_names = parameter_names
        else:
            self._parameter_names = list(self._data.columns)

    def __str__(self):
        return self._data.__str__()

    @property
    def parameter_names(self):
        # type: () -> List[str]
        return self._parameter_names

    @property
    def values(self):
        # type: () -> np.ndarray
        # Cache the transposed array to avoid recreating it on each call
        if not hasattr(self, '_cached_values_array') or self._cached_values_array is None:
            self._cached_values_array = np.array(self._data_numeric).T
        return self._cached_values_array

    def clear(self):
        self.data = pd.DataFrame()

    def compute_columns(
            self,
            constants,  # type: Dict[str, float]
            equations,  # type: List[Dict[str, str]]
            equation_json_fn=None  # type: str
    ):
        compute_values(
            d=self.data,
            constants=constants,
            equations=equations,
            equation_json_fn=equation_json_fn
        )
        # use property to update dependent attributes
        self.data = self.data

    @property
    def empty(self):
        # type: ()->(bool)
        return self._data.empty

    def get_mask(
            self,
            selections,  # type: List[DataSelection]
            idxs=None,  # type: List[int]
            mask_nan=True,  # type: bool
            mask_inf=True  # type: bool
    ):
        """
        :param selections: A list of selections
        :param idxs: Which parameter indices to check for NaN/Inf
        :param mask_nan: If True, NaN values are masked
        :param mask_inf: If True, Inf values are masked
        :return: A boolean array 'mask' of shape (n_parameter, n_data_points),
                 where True indicates the value is masked.
        """
        if idxs is None:
            idxs = []

        d = self.values  # shape: (n_parameter, n_data_points)
        n_parameter, n_data_points = d.shape

        # Pre-allocate a boolean mask, all set to False initially
        mask = np.zeros((n_parameter, n_data_points), dtype=bool)

        # 1) Apply each DataSelection
        for sel in selections:
            if not sel.enabled:
                continue
            if sel.parameter_idx >= n_parameter:
                # skip invalid indices
                continue

            param_data = d[sel.parameter_idx, :]  # shape: (n_data_points,)

            if sel.invert:
                # Mask where data is within (lower, upper)
                out_of_bounds = (param_data > sel.lower) & (param_data < sel.upper)
            else:
                # Mask where data is outside [lower, upper]
                out_of_bounds = (param_data < sel.lower) | (param_data > sel.upper)

            # For all data points that are out_of_bounds, mask across *all parameters*
            # (i.e., set True in the entire row for those columns).
            mask[:, out_of_bounds] = True

        # 2) Mask NaN/Inf in specified idxs
        for idx in idxs:
            if idx < 0 or idx >= n_parameter:
                continue

            col_data = d[idx, :]  # shape: (n_data_points,)

            # Combine NaN and Inf checks if both are needed
            bad_vals = np.zeros(n_data_points, dtype=bool)

            if mask_nan:
                bad_vals |= np.isnan(col_data)
            if mask_inf:
                bad_vals |= np.isinf(col_data)

            # Mask entire row for those columns
            mask[:, bad_vals] = True

        return mask

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, v):
        self._data = v
        self._parameter_names = list(self._data.columns)
        self._data_numeric = v.apply(pd.to_numeric, errors='coerce')
        # Invalidate the cached values array when data changes
        if hasattr(self, '_cached_values_array'):
            self._cached_values_array = None

    @property
    def size(self):
        return self.values.shape[1]

    def merge(self, other_source, mode='columns'):
        """
        Merge data from another DataSource into this one.

        :param other_source: Another DataSource object to merge with this one
        :param mode: How to merge the data - 'columns' or 'rows'
                    'columns': Add all columns that don't already exist, check that row count matches
                    'rows': Append rows of existing columns, show warning if there are new columns
        :return: True if merge was successful, False otherwise
        """
        from PyQt5.QtWidgets import QMessageBox

        if mode == 'columns':
            # Check if row count matches
            if len(self.data) != len(other_source.data):
                QMessageBox.warning(
                    None, 
                    "Row Count Mismatch",
                    f"New data has {len(other_source.data)} rows, but current data has {len(self.data)} rows. Data will not be merged."
                )
                return False

            # Identify duplicate columns
            duplicate_cols = set(self.data.columns).intersection(set(other_source.data.columns))

            # Remove duplicate columns from the new DataFrame
            df_unique = other_source.data.drop(columns=duplicate_cols)

            # Combine with the existing DataFrame
            combined_df = pd.concat([self.data, df_unique], axis=1)

            # Update this DataSource with the combined data
            self.data = combined_df

            return True

        elif mode == 'rows':
            # When appending rows, we need to handle columns differently

            # Get the set of columns in both DataFrames
            existing_cols = set(self.data.columns)
            new_cols = set(other_source.data.columns)

            # Check if there are new columns in the other source
            new_unique_cols = new_cols - existing_cols
            if new_unique_cols:
                QMessageBox.warning(
                    None,
                    "New Columns Found",
                    f"New data contains columns not in the current data: {', '.join(new_unique_cols)}. "
                    f"Only rows of existing columns will be appended."
                )

            # Get common columns to append
            common_cols = existing_cols.intersection(new_cols)

            if not common_cols:
                QMessageBox.warning(
                    None,
                    "No Common Columns",
                    "New data has no columns in common with the current data. Cannot append rows."
                )
                return False

            # Create a new DataFrame with only the common columns from the other source
            other_common_df = other_source.data[list(common_cols)]

            # Append the rows
            combined_df = pd.concat([self.data, other_common_df], axis=0, ignore_index=True)

            # Update this DataSource with the combined data
            self.data = combined_df

            return True

        else:
            QMessageBox.warning(
                None,
                "Invalid Merge Mode",
                f"Invalid merge mode: {mode}. Must be 'columns' or 'rows'."
            )
            return False
