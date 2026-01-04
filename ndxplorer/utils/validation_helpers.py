"""Helper utilities for common validation patterns."""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple, Union

import numpy as np

from ..logging_config import logging


def is_valid_histogram(hist: Any) -> bool:
    """Check if histogram is in valid format (edges, counts) or (H, x_edges, y_edges)."""
    if hist is None:
        return False
    if isinstance(hist, tuple) and len(hist) == 2:
        edges, counts = hist
        try:
            return len(edges) >= 2 and len(counts) >= 1
        except (TypeError, AttributeError):
            return False
    if isinstance(hist, tuple) and len(hist) == 3:
        H, x_edges, y_edges = hist
        try:
            return H.size > 0 and len(x_edges) >= 2 and len(y_edges) >= 2
        except (TypeError, AttributeError):
            return False
    return False


def is_valid_bins(bins: Any) -> bool:
    """Return True when bins is a strictly increasing 1D array."""
    try:
        if bins is None:
            return False
        arr = np.asarray(bins)
        if arr.ndim != 1 or arr.size < 2:
            return False
        return np.all(np.diff(arr) > 0)
    except Exception:
        return False


def is_valid_range(value: Any, min_val: Optional[float] = None, max_val: Optional[float] = None) -> bool:
    """Check if value is within specified range."""
    try:
        if not isinstance(value, (int, float, np.number)):
            return False
        if not np.isfinite(value):
            return False
        if min_val is not None and value < min_val:
            return False
        if max_val is not None and value > max_val:
            return False
        return True
    except Exception:
        return False


def is_valid_array(arr: Any, ndim: Optional[int] = None, min_size: int = 0) -> bool:
    """Check if array is valid with optional dimension and size constraints."""
    try:
        if arr is None:
            return False
        arr = np.asarray(arr)
        if ndim is not None and arr.ndim != ndim:
            return False
        if arr.size < min_size:
            return False
        return True
    except Exception:
        return False


def validate_numeric_value(
    value: Any,
    default: Union[int, float] = 0,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    allow_none: bool = False,
) -> Union[int, float, None]:
    """Validate and convert value to numeric, returning default if invalid."""
    if value is None:
        return None if allow_none else default
    try:
        num_value = float(value) if not isinstance(value, (int, float)) else value
        if not np.isfinite(num_value):
            return default
        if min_val is not None and num_value < min_val:
            return default
        if max_val is not None and num_value > max_val:
            return default
        return num_value
    except (TypeError, ValueError):
        return default


def validate_string_choice(
    value: str,
    choices: Tuple[str, ...],
    default: str = None,
    case_sensitive: bool = False,
) -> str:
    """Validate string is one of allowed choices."""
    if value is None:
        return default
    try:
        test_value = value if case_sensitive else str(value).lower()
        test_choices = choices if case_sensitive else tuple(c.lower() for c in choices)
        if test_value in test_choices:
            return value
        return default
    except Exception:
        return default


def validate_shape(
    arr: np.ndarray,
    expected_shape: Tuple[int, ...],
    allow_broadcast: bool = False,
) -> bool:
    """Validate array shape matches expected shape."""
    try:
        arr = np.asarray(arr)
        if arr.shape == expected_shape:
            return True
        if allow_broadcast:
            # Check if shapes are broadcastable
            try:
                np.broadcast_shapes(arr.shape, expected_shape)
                return True
            except ValueError:
                return False
        return False
    except Exception:
        return False


def validate_data_consistency(
    data: np.ndarray,
    edges: np.ndarray,
    axis: int = 0,
) -> bool:
    """Validate histogram data and edges are consistent."""
    try:
        data = np.asarray(data)
        edges = np.asarray(edges)
        
        # Check edges are 1D and strictly increasing
        if edges.ndim != 1 or edges.size < 2:
            return False
        if not np.all(np.diff(edges) > 0):
            return False
        
        # Check data size matches edges
        expected_size = edges.size - 1
        if data.ndim == 1:
            return data.size == expected_size
        else:
            return data.shape[axis] == expected_size
    except Exception:
        return False
