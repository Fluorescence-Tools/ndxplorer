"""Helper utilities for consistent exception handling and logging."""

from __future__ import annotations

from typing import Callable, TypeVar, Optional, Any
from functools import wraps

from ..logging_config import logging

T = TypeVar('T')


def safe_call(func: Callable[..., T], *args, default: T = None, log_errors: bool = True, **kwargs) -> T:
    """Safely call a function with exception handling and optional logging."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logging.debug(f"Error calling {func.__name__}: {e}")
        return default


def safe_execute(func: Callable, *args, log_level: str = 'debug', **kwargs) -> bool:
    """Execute a function safely and return success status."""
    try:
        func(*args, **kwargs)
        return True
    except Exception as e:
        log_fn = getattr(logging, log_level, logging.debug)
        log_fn(f"Error executing {func.__name__}: {e}")
        return False


def handle_exception(log_level: str = 'warning', return_value: T = None, reraise: bool = False):
    """Decorator for consistent exception handling."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log_fn = getattr(logging, log_level, logging.warning)
                log_fn(f"Error in {func.__name__}: {e}")
                if reraise:
                    raise
                return return_value
        return wrapper
    return decorator


def log_exception(exc: Exception, context: str = "", log_level: str = 'warning', exc_info: bool = False) -> None:
    """Log an exception with context."""
    log_fn = getattr(logging, log_level, logging.warning)
    if context:
        log_fn(f"{context}: {exc}", exc_info=exc_info)
    else:
        log_fn(f"Exception: {exc}", exc_info=exc_info)


def try_get_attribute(obj: Any, attr: str, default: T = None, log_errors: bool = False) -> T:
    """Safely get attribute with optional error logging."""
    try:
        return getattr(obj, attr)
    except AttributeError as e:
        if log_errors:
            logging.debug(f"Attribute {attr} not found on {type(obj).__name__}: {e}")
        return default
