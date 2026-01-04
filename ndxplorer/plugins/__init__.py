"""Plugin namespace for optional integrations."""

from .napari_viewer import (
    FEATURE_FLAG_ENV,
    get_or_create_viewer,
    is_enabled,
    is_available,
    show_dataset,
)

__all__ = [
    "FEATURE_FLAG_ENV",
    "get_or_create_viewer",
    "is_enabled",
    "is_available",
    "show_dataset",
]
