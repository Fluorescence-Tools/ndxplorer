"""
Data management layer for NDXplorer.

This package provides a clean separation between data operations and UI,
making the code more testable and maintainable.
"""

from .data_manager import DataManager
from .cache_coordinator import CacheCoordinator
from .selection_manager import SelectionManager

__all__ = ['DataManager', 'CacheCoordinator', 'SelectionManager']
