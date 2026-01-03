"""
Cache coordination for NDXplorer data operations.

Manages caching of computed values, histograms, and other expensive operations.
"""

from typing import Any, Optional, Dict
from ...logging_config import logging


class CacheCoordinator:
    """
    Coordinates caching of computed values and expensive operations.
    
    Provides simple key-value caching with invalidation support.
    """
    
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        
    def get_cache_value(self, key: str) -> Optional[Any]:
        """
        Retrieve a cached value.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        return self._cache.get(key)
        
    def set_cache_value(self, key: str, value: Any) -> None:
        """
        Store a value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        self._cache[key] = value
        logging.debug(f"CacheCoordinator: Cached {key}")
        
    def invalidate(self, key: str) -> None:
        """
        Invalidate a specific cache entry.
        
        Args:
            key: Cache key to invalidate
        """
        if key in self._cache:
            del self._cache[key]
            logging.debug(f"CacheCoordinator: Invalidated {key}")
            
    def invalidate_all(self) -> None:
        """Invalidate all cached values."""
        self._cache.clear()
        logging.debug("CacheCoordinator: Invalidated all cache")
        
    def get_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        return {
            'num_entries': len(self._cache),
            'keys': list(self._cache.keys())
        }
