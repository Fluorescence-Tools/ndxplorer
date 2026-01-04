"""
NDXplorer package init with optional GUI-heavy imports.
"""

from __future__ import annotations

from typing import Optional

NDXplorer: Optional[object]
FixedImageItem: Optional[object]
_NDX_IMPORT_ERROR: Optional[Exception] = None
_IMAGE_IMPORT_ERROR: Optional[Exception] = None

try:  # pragma: no cover - optional GUI dependency
    from .core.plot_main import NDXplorer as _NDXplorer  # type: ignore
    NDXplorer = _NDXplorer
except ModuleNotFoundError as exc:  # pragma: no cover - fallback path
    NDXplorer = None
    _NDX_IMPORT_ERROR = exc

try:  # pragma: no cover - optional GUI dependency
    from .plotting.image_items import FixedImageItem as _FixedImageItem  # type: ignore
    FixedImageItem = _FixedImageItem
except ModuleNotFoundError as exc:  # pragma: no cover - fallback path
    FixedImageItem = None
    _IMAGE_IMPORT_ERROR = exc


def __getattr__(name: str):
    if name == "NDXplorer":
        if NDXplorer is None and _NDX_IMPORT_ERROR is not None:
            raise ModuleNotFoundError(
                "NDXplorer GUI components require optional dependencies (e.g., pyqtgraph)."
            ) from _NDX_IMPORT_ERROR
        return NDXplorer
    if name == "FixedImageItem":
        if FixedImageItem is None and _IMAGE_IMPORT_ERROR is not None:
            raise ModuleNotFoundError(
                "FixedImageItem requires optional GUI dependencies."
            ) from _IMAGE_IMPORT_ERROR
        return FixedImageItem
    raise AttributeError(name)


__all__ = ["NDXplorer", "FixedImageItem"]
