"""
Performance configuration and optimization settings for ndxplorer.

Provides centralized control over performance features:
- Bitfield masks
- Histogram caching
- Numba acceleration
- Memory limits
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from ..logging_config import logging


@dataclass
class PerformanceConfig:
    """
    Performance optimization configuration.
    
    Attributes
    ----------
    use_bitfield_masks : bool
        Use bitfield masks for 8x memory savings (default: auto-detect based on data size)
    use_histogram_cache : bool
        Cache histogram computations (default: True)
    use_boost_histogram : bool
        Use boost-histogram library when available (fastest, default: True)
    use_numba : bool
        Use Numba JIT compilation when available (default: True)
    use_fast_histogram : bool
        Use optimized histogram implementation (default: True)
    histogram_cache_memory_mb : float
        Memory limit for histogram cache in MB (default: 200)
    general_cache_memory_mb : float
        Memory limit for general cache in MB (default: 50)
    bitfield_threshold : int
        Minimum number of points to use bitfield masks (default: 100000)
    parallel_histogram : bool
        Use parallel histogram computation (requires Numba, default: True)
    aggressive_caching : bool
        Enable aggressive caching for all operations (default: True)
    histogram_threads : int
        Number of threads for boost-histogram (-1 for auto-detect, 0 or 1 for single-threaded, default: -1)
    plot_backend : str
        Plotting backend to use ('guiqwt', 'matplotlib', 'pyqtgraph', default: 'guiqwt')
    """
    
    use_bitfield_masks: bool = True
    use_histogram_cache: bool = True
    use_boost_histogram: bool = True
    use_numba: bool = True
    use_fast_histogram: bool = True
    histogram_cache_memory_mb: float = 200.0
    general_cache_memory_mb: float = 50.0
    bitfield_threshold: int = 100000
    parallel_histogram: bool = True
    aggressive_caching: bool = True
    histogram_threads: int = -1
    plot_backend: str = "guiqwt"
    
    @classmethod
    def from_environment(cls) -> "PerformanceConfig":
        """Create configuration from environment variables with settings file overrides."""
        return cls(
            use_bitfield_masks=_get_env_with_settings_override("NDXPLORER_USE_BITFIELD", True),
            use_histogram_cache=_get_env_with_settings_override("NDXPLORER_USE_HISTOGRAM_CACHE", True),
            use_boost_histogram=_get_env_with_settings_override("NDXPLORER_USE_BOOST_HISTOGRAM", True),
            use_numba=_get_env_with_settings_override("NDXPLORER_USE_NUMBA", True),
            use_fast_histogram=_get_env_with_settings_override("NDXPLORER_USE_FAST_HISTOGRAM", True),
            histogram_cache_memory_mb=_get_float_env_with_settings_override("NDXPLORER_HISTOGRAM_CACHE_MB", 200.0),
            general_cache_memory_mb=_get_float_env_with_settings_override("NDXPLORER_GENERAL_CACHE_MB", 50.0),
            bitfield_threshold=_get_int_env_with_settings_override("NDXPLORER_BITFIELD_THRESHOLD", 100000),
            parallel_histogram=_get_env_with_settings_override("NDXPLORER_PARALLEL_HISTOGRAM", True),
            aggressive_caching=_get_env_with_settings_override("NDXPLORER_AGGRESSIVE_CACHING", True),
            histogram_threads=_get_int_env_with_settings_override("NDXPLORER_HISTOGRAM_THREADS", -1),
            plot_backend=_get_str_env_with_settings_override("NDXPLORER_PLOT_BACKEND", "guiqwt"),
        )
    
    @classmethod
    def high_performance(cls) -> "PerformanceConfig":
        """Configuration optimized for maximum speed."""
        return cls(
            use_bitfield_masks=True,
            use_histogram_cache=True,
            use_boost_histogram=True,
            use_numba=True,
            use_fast_histogram=True,
            histogram_cache_memory_mb=500.0,
            general_cache_memory_mb=100.0,
            bitfield_threshold=50000,
            parallel_histogram=True,
            aggressive_caching=True,
            histogram_threads=-1,
            plot_backend="guiqwt",
        )
    
    @classmethod
    def low_memory(cls) -> "PerformanceConfig":
        """Configuration optimized for low memory usage."""
        return cls(
            use_bitfield_masks=True,
            use_histogram_cache=True,
            use_boost_histogram=True,
            use_numba=True,
            use_fast_histogram=True,
            histogram_cache_memory_mb=50.0,
            general_cache_memory_mb=20.0,
            bitfield_threshold=200000,
            parallel_histogram=False,
            aggressive_caching=False,
            histogram_threads=1,
            plot_backend="guiqwt",
        )
    
    @classmethod
    def balanced(cls) -> "PerformanceConfig":
        """Balanced configuration (default)."""
        return cls()
    
    def log_config(self) -> None:
        """Log current configuration."""
        logging.info("[PerformanceConfig] Active settings:")
        logging.info(f"  Bitfield masks: {self.use_bitfield_masks} (threshold: {self.bitfield_threshold})")
        logging.info(f"  Histogram cache: {self.use_histogram_cache} ({self.histogram_cache_memory_mb} MB)")
        logging.info(f"  Boost-histogram: {self.use_boost_histogram} (threads: {self.histogram_threads})")
        logging.info(f"  Numba acceleration: {self.use_numba}")
        logging.info(f"  Fast histogram: {self.use_fast_histogram}")
        logging.info(f"  Parallel histogram: {self.parallel_histogram}")
        logging.info(f"  Aggressive caching: {self.aggressive_caching}")
        logging.info(f"  Plot backend: {self.plot_backend}")


def _get_env_with_settings_override(key: str, default: bool) -> bool:
    """Get boolean from environment variable with settings file override."""
    # First check if there's a settings file override
    settings_env = _get_environment_overrides()
    if key in settings_env and settings_env[key] is not None:
        value = str(settings_env[key]).lower()
        if value in ("1", "true", "yes", "on"):
            return True
        elif value in ("0", "false", "no", "off"):
            return False
        else:
            logging.warning(f"Invalid boolean value for {key} in settings: {settings_env[key]}")
    
    # Fall back to environment variable
    return _get_bool_env(key, default)


def _get_float_env_with_settings_override(key: str, default: float) -> float:
    """Get float from environment variable with settings file override."""
    # First check if there's a settings file override
    settings_env = _get_environment_overrides()
    if key in settings_env and settings_env[key] is not None:
        try:
            return float(settings_env[key])
        except (ValueError, TypeError):
            logging.warning(f"Invalid float value for {key} in settings: {settings_env[key]}")
    
    # Fall back to environment variable
    return _get_float_env(key, default)


def _get_int_env_with_settings_override(key: str, default: int) -> int:
    """Get integer from environment variable with settings file override."""
    # First check if there's a settings file override
    settings_env = _get_environment_overrides()
    if key in settings_env and settings_env[key] is not None:
        try:
            return int(settings_env[key])
        except (ValueError, TypeError):
            logging.warning(f"Invalid integer value for {key} in settings: {settings_env[key]}")
    
    # Fall back to environment variable
    return _get_int_env(key, default)


def _get_str_env_with_settings_override(key: str, default: str) -> str:
    """Get string from environment variable with settings file override."""
    # First check if there's a settings file override
    settings_env = _get_environment_overrides()
    if key in settings_env and settings_env[key] is not None:
        value = str(settings_env[key]).strip()
        if value:  # Check if not empty
            return value
        else:
            logging.warning(f"Empty string value for {key} in settings")
    
    # Fall back to environment variable
    return os.environ.get(key, default)


def _get_environment_overrides() -> dict:
    """Load environment variable overrides from settings file."""
    global _environment_overrides_cache
    if _environment_overrides_cache is not None:
        return _environment_overrides_cache
    
    overrides = {}
    try:
        # Try to find and load the settings file
        settings_paths = [
            # User settings directory
            Path.home() / ".chisurf" / "ndxplorer" / "mfd.settings.json",
            # Module settings directory
            Path(__file__).parent.parent / "settings" / "mfd.settings.json",
            # Current directory
            Path.cwd() / "mfd.settings.json",
        ]
        
        for settings_path in settings_paths:
            if settings_path.exists():
                with open(settings_path, 'r', encoding='utf-8') as f:
                    settings_data = json.load(f)
                
                if 'environment' in settings_data:
                    overrides = settings_data['environment']
                    logging.debug(f"Loaded {len(overrides)} environment overrides from {settings_path}")
                    break
        
        if not overrides:
            logging.debug("No environment overrides found in settings files")
            
    except Exception as e:
        logging.warning(f"Failed to load environment overrides from settings: {e}")
        overrides = {}
    
    _environment_overrides_cache = overrides
    return overrides


# Global cache for environment overrides
_environment_overrides_cache: Optional[dict] = None


def _get_bool_env(key: str, default: bool) -> bool:
    """Get boolean from environment variable."""
    value = os.environ.get(key, "").lower()
    if value in ("1", "true", "yes", "on"):
        return True
    elif value in ("0", "false", "no", "off"):
        return False
    return default


def _get_float_env(key: str, default: float) -> float:
    """Get float from environment variable."""
    value = os.environ.get(key)
    if value is not None:
        try:
            return float(value)
        except ValueError:
            pass
    return default


def _get_int_env(key: str, default: int) -> int:
    """Get integer from environment variable."""
    value = os.environ.get(key)
    if value is not None:
        try:
            return int(value)
        except ValueError:
            pass
    return default


# Global configuration instance
_global_config: Optional[PerformanceConfig] = None


def get_performance_config() -> PerformanceConfig:
    """Get or create global performance configuration."""
    global _global_config
    if _global_config is None:
        _global_config = PerformanceConfig.from_environment()
        _global_config.log_config()
    return _global_config


def set_performance_config(config: PerformanceConfig) -> None:
    """Set global performance configuration."""
    global _global_config
    _global_config = config
    config.log_config()


def reset_performance_config() -> None:
    """Reset to default configuration."""
    global _global_config, _environment_overrides_cache
    _global_config = None
    _environment_overrides_cache = None


# Convenience functions for common configurations

def enable_high_performance() -> None:
    """Enable high-performance mode."""
    set_performance_config(PerformanceConfig.high_performance())
    logging.info("[Performance] High-performance mode enabled")


def enable_low_memory() -> None:
    """Enable low-memory mode."""
    set_performance_config(PerformanceConfig.low_memory())
    logging.info("[Performance] Low-memory mode enabled")


def enable_balanced() -> None:
    """Enable balanced mode (default)."""
    set_performance_config(PerformanceConfig.balanced())
    logging.info("[Performance] Balanced mode enabled")
