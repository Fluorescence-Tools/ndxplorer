"""Image export utilities for NDXplorer selections."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..logging_config import logging
from ..utils.performance_optimizations import get_performance_monitor
from .models import SelectionExportPayload


def export_image(
    payload: SelectionExportPayload,
    path: Path,
    *,
    dpi: int = 300,
    transparent: bool = False,
    quality: int | None = None,
) -> None:
    """
    Persist the current selection visualization as an image.

    Supports Matplotlib figures, Qt-compatible QImage/QPixmap, and any object
    exposing a ``save(path, format=...)`` signature.
    """

    source = _resolve_source(payload)
    path.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Exporting image to %s", path)
    perf = get_performance_monitor()
    op_name = f"export_image[{path.suffix.lower() or 'unknown'}]"
    perf.start_timer(op_name)
    try:
        if hasattr(source, "savefig"):
            _save_matplotlib(source, path, dpi=dpi, transparent=transparent)
        elif hasattr(source, "save"):
            _save_via_save_method(source, path, quality=quality)
        else:
            raise ValueError("Image export requires figure/image/pixmap with save capability.")
    finally:
        perf.end_timer(op_name)
        perf.log_memory_usage(op_name)


def _resolve_source(payload: SelectionExportPayload) -> Any:
    if payload.figure is not None:
        return payload.figure
    if payload.image is not None:
        return payload.image
    if payload.pixmap is not None:
        return payload.pixmap
    raise ValueError("Selection payload does not contain any drawable object.")


def _save_matplotlib(fig: Any, path: Path, *, dpi: int, transparent: bool) -> None:
    fig.savefig(path, dpi=dpi, transparent=transparent, bbox_inches="tight")


def _save_via_save_method(obj: Any, path: Path, *, quality: int | None) -> None:
    kwargs = {}
    suffix = path.suffix.lower().lstrip(".")
    if quality is not None and suffix in {"jpg", "jpeg"}:
        kwargs["quality"] = quality
    if not obj.save(str(path), **kwargs):
        raise RuntimeError(f"Failed to save image to {path}")


__all__ = ["export_image"]
