"""File metadata caching for NDXplorer to avoid re-detection overhead."""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Dict, Optional, Tuple
import time

try:
    from ..logging_config import logging
except Exception:  # pragma: no cover
    import logging  # type: ignore


class FileMetadataCache:
    """Cache for file format metadata to avoid re-detection on subsequent loads."""
    
    def __init__(self, cache_dir: Optional[pathlib.Path] = None):
        """
        Initialize the metadata cache.
        
        Parameters
        ----------
        cache_dir : Optional[pathlib.Path]
            Directory to store cache files. If None, uses temp directory.
        """
        if cache_dir is None:
            import tempfile
            cache_dir = pathlib.Path(tempfile.gettempdir()) / "ndxplorer_cache"
        
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "file_metadata.json"
        
        # Load existing cache
        self._cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """Load existing cache from disk."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logging.debug(f"Failed to load metadata cache: {e}")
        return {}
    
    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to save metadata cache: {e}")
    
    def _get_file_hash(self, file_path: pathlib.Path) -> str:
        """Get a hash of file metadata for cache key."""
        try:
            stat = file_path.stat()
            # Use file size and modification time for quick hashing
            hash_data = f"{file_path}_{stat.st_size}_{stat.st_mtime}"
            return hashlib.md5(hash_data.encode()).hexdigest()
        except Exception:
            return hashlib.md5(str(file_path).encode()).hexdigest()
    
    def get_cached_format(self, file_path: pathlib.Path) -> Optional[Dict]:
        """
        Get cached format information for a file.
        
        Parameters
        ----------
        file_path : pathlib.Path
            Path to the file
            
        Returns
        -------
        Optional[Dict]
            Cached format kwargs or None if not cached/invalid
        """
        file_hash = self._get_file_hash(file_path)
        cache_key = str(file_path)
        
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if entry.get('hash') == file_hash:
                logging.debug(f"Using cached format for {file_path.name}")
                return entry.get('format')
            else:
                # File changed, remove old entry
                del self._cache[cache_key]
                logging.debug(f"File changed, invalidating cache for {file_path.name}")
        
        return None
    
    def cache_format(self, file_path: pathlib.Path, format_kwargs: Dict) -> None:
        """
        Cache format information for a file.
        
        Parameters
        ----------
        file_path : pathlib.Path
            Path to the file
        format_kwargs : Dict
            Format kwargs to cache
        """
        cache_key = str(file_path)
        file_hash = self._get_file_hash(file_path)
        
        self._cache[cache_key] = {
            'hash': file_hash,
            'format': format_kwargs,
            'timestamp': time.time()
        }
        
        # Limit cache size to prevent unbounded growth
        if len(self._cache) > 1000:
            # Remove oldest entries
            sorted_items = sorted(self._cache.items(), key=lambda x: x[1]['timestamp'])
            for key, _ in sorted_items[:100]:  # Remove oldest 100 entries
                del self._cache[key]
        
        self._save_cache()
        logging.debug(f"Cached format for {file_path.name}")
    
    def clear_cache(self) -> None:
        """Clear all cached metadata."""
        self._cache.clear()
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
        except Exception as e:
            logging.warning(f"Failed to delete cache file: {e}")


# Global cache instance
_metadata_cache: Optional[FileMetadataCache] = None


def get_metadata_cache() -> FileMetadataCache:
    """Get the global metadata cache instance."""
    global _metadata_cache
    if _metadata_cache is None:
        _metadata_cache = FileMetadataCache()
    return _metadata_cache


def clear_metadata_cache() -> None:
    """Clear the global metadata cache."""
    global _metadata_cache
    if _metadata_cache is not None:
        _metadata_cache.clear_cache()
        _metadata_cache = None
