"""Helper utilities for safe attribute access and common patterns."""

from __future__ import annotations

from typing import Any, Optional, TypeVar, Callable

T = TypeVar('T')


def safe_getattr(obj: Any, attr: str, default: T = None) -> T:
    """Safely get attribute with default value."""
    return getattr(obj, attr, default)


def safe_hasattr_and_get(obj: Any, attr: str, default: T = None) -> T:
    """Check if attribute exists and get it, otherwise return default."""
    if hasattr(obj, attr):
        return getattr(obj, attr)
    return default


def safe_call_method(obj: Any, method_name: str, *args, **kwargs) -> Any:
    """Safely call a method if it exists."""
    method = safe_getattr(obj, method_name)
    if callable(method):
        try:
            return method(*args, **kwargs)
        except Exception:
            return None
    return None


def safe_check_and_get(obj: Any, attr: str, check_fn: Callable = None, default: T = None) -> T:
    """Check attribute exists, optionally validate with check_fn, and get it."""
    if not hasattr(obj, attr):
        return default
    value = getattr(obj, attr)
    if check_fn is not None:
        try:
            if not check_fn(value):
                return default
        except Exception:
            return default
    return value


def safe_get_nested(obj: Any, path: str, default: T = None) -> T:
    """Safely get nested attribute using dot notation (e.g., 'parent.child.attr')."""
    try:
        current = obj
        for attr in path.split('.'):
            current = getattr(current, attr)
        return current
    except (AttributeError, TypeError):
        return default


def has_all_attrs(obj: Any, *attrs: str) -> bool:
    """Check if object has all specified attributes."""
    return all(hasattr(obj, attr) for attr in attrs)


def has_any_attrs(obj: Any, *attrs: str) -> bool:
    """Check if object has any of the specified attributes."""
    return any(hasattr(obj, attr) for attr in attrs)


def get_attrs(obj: Any, *attrs: str, default: Any = None) -> dict:
    """Get multiple attributes as a dictionary, using default for missing ones."""
    return {attr: getattr(obj, attr, default) for attr in attrs}


def safe_get_and_call(obj: Any, method_name: str, *args, default: Any = None, **kwargs) -> Any:
    """Safely get and call a method, returning default if it fails."""
    try:
        method = getattr(obj, method_name, None)
        if callable(method):
            return method(*args, **kwargs)
        return default
    except Exception:
        return default
