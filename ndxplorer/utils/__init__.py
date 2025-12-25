"""Utility helpers with optional GUI dependencies."""

from __future__ import annotations

import warnings

from .axis_helpers import *
from .histogram_helpers import *
from .histogram_export import *
from .napari_helpers import *
from .screenshot_helpers import *
from .ui_helpers import *
from .working_path_helpers import *
from .lazy_imports import *
from .performance_optimizations import *
from .value_cache import *
from .mouse_event_filter import MouseEventFilter
from .. import settings_helpers as settings_helpers

try:  # pragma: no cover - optional GUI dependency
    from .colormap_helpers import *
except ModuleNotFoundError as exc:  # pragma: no cover - warn but continue
    warnings.warn(
        f"ndxplorer.utils.colormap_helpers unavailable: {exc}. "
        "Colormap customization features may be disabled.",
        RuntimeWarning,
    )

__all__ = ["MouseEventFilter"]
