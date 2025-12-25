"""Optional Napari viewer bridge for NDXplorer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import os

from qtpy import QtWidgets

from ..logging_config import logging
from ..utils import napari_helpers

FEATURE_FLAG_ENV = "NDX_ENABLE_NAPARI"


@dataclass
class NapariStatus:
    enabled: bool
    available: bool
    reason: Optional[str] = None


def _feature_enabled() -> bool:
    """Return True when the feature flag enables Napari integration."""
    value = os.environ.get(FEATURE_FLAG_ENV, "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return False


def is_enabled() -> bool:
    """Public helper for feature flag state."""
    return _feature_enabled()


def is_available(ndxplorer: "NDXplorer") -> bool:
    """Check if Napari can be used (flag + module presence)."""
    if not is_enabled():
        logging.debug("Napari plugin disabled via feature flag %s", FEATURE_FLAG_ENV)
        return False
    return napari_helpers.is_napari_available(ndxplorer)


def ensure_available(ndxplorer: "NDXplorer") -> bool:
    """Ensure feature flag and dependency readiness."""
    if not is_enabled():
        QtWidgets.QMessageBox.information(
            ndxplorer,
            "Napari Disabled",
            (
                "Set the environment variable NDX_ENABLE_NAPARI=1 before launching ChiSurf\n"
                "to enable the optional Napari viewer integration."
            ),
        )
        return False
    return napari_helpers.ensure_napari_available(ndxplorer)


def get_or_create_viewer(ndxplorer: "NDXplorer"):
    """Return an existing Napari viewer or create a new one."""
    if not ensure_available(ndxplorer):
        return None
    return napari_helpers.get_or_create_viewer(ndxplorer)


def show_dataset(ndxplorer: "NDXplorer", data: Any, *, metadata: Optional[dict[str, Any]] = None):
    """Generic entry point for sending data objects to Napari."""
    if not ensure_available(ndxplorer):
        return

    viewer = get_or_create_viewer(ndxplorer)
    if viewer is None:
        logging.debug("Napari viewer could not be obtained")
        return

    layer_kwargs = metadata or {}
    try:
        if isinstance(data, dict):
            image = data.get("image")
            if image is None:
                raise ValueError("data dict must contain 'image'")
            viewer.add_image(image, **layer_kwargs)
        else:
            viewer.add_image(data, **layer_kwargs)
    except Exception as exc:
        logging.error("Failed to show dataset in Napari: %s", exc)


def send_histogram(ndxplorer: "NDXplorer"):
    """Convenience layer replicating legacy send_to_napari behavior."""
    if not ensure_available(ndxplorer):
        return
    napari_helpers.send_to_napari(ndxplorer)
