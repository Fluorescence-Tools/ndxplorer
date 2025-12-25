"""Public export API for NDXplorer."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Union

from . import csv_export, image_export, hdf5_export
from .manifest import write_manifest
from .models import (
    SelectionExportPayload,
    ExportValidationError,
    validate_payload_integrity,
)
from .path_utils import normalize_export_path


SelectionInput = Union[SelectionExportPayload, Mapping[str, object]]

SUPPORTED_FORMATS: Mapping[str, Sequence[str]] = {
    "csv": (".csv", ".tsv", ".txt"),
    "image": (".png", ".jpg", ".jpeg", ".svg"),
    "hdf5": (".h5", ".hdf5"),
}


def list_supported_formats() -> Mapping[str, Sequence[str]]:
    """Return mapping of export families to supported file suffixes."""
    return SUPPORTED_FORMATS


def _normalize_payload(payload: SelectionInput) -> SelectionExportPayload:
    if isinstance(payload, SelectionExportPayload):
        return payload

    return SelectionExportPayload(
        selections=payload.get("selections", ()),
        table=payload.get("table"),
        values=payload.get("values"),
        columns=payload.get("columns"),
        figure=payload.get("figure"),
        image=payload.get("image"),
        pixmap=payload.get("pixmap"),
        metadata=payload.get("metadata", {}),
        name=payload.get("name"),
    )


def _infer_family(path: Union[str, Path], explicit_format: Optional[str]) -> str:
    if explicit_format:
        normalized = explicit_format.lower()
        if normalized in SUPPORTED_FORMATS:
            return normalized
        raise ValueError(f"Unsupported export format '{explicit_format}'.")

    suffix = Path(path).suffix.lower()
    for family, suffixes in SUPPORTED_FORMATS.items():
        if suffix in suffixes:
            return family
    raise ValueError(f"Cannot infer export family from suffix '{suffix}' – please pass format=...")


def _validate_payload_for_family(family: str, payload: SelectionExportPayload) -> None:
    if family in {"csv", "hdf5"} and not payload.has_tabular_data():
        raise ExportValidationError(
            f"Export format '{family}' requires tabular selection data (table or values)."
        )
    if family == "image" and not payload.has_drawable():
        raise ExportValidationError(
            "Image export requires a figure/image/pixmap on the selection payload."
        )


def save_selection(
    payload: SelectionInput,
    path: Union[str, Path],
    *,
    format: Optional[str] = None,
    options: Optional[Mapping[str, object]] = None,
    write_manifest_file: bool = True,
) -> Path:
    """
    Persist a selection payload in the requested format.

    Parameters
    ----------
    payload:
        SelectionExportPayload or dict-like structure.
    path:
        Destination file path.
    format:
        One of ``'csv'``, ``'image'``, ``'hdf5'``. If omitted, inferred from suffix.
    options:
        Extra backend-specific options (delimiter, dpi, compression, ...).
    """

    target = normalize_export_path(path)
    export_options = dict(options or {})
    normalized_payload = _normalize_payload(payload)
    validate_payload_integrity(normalized_payload)
    family = _infer_family(target, format)
    _validate_payload_for_family(family, normalized_payload)

    if family == "csv":
        csv_export.export_table(normalized_payload, target, **export_options)
    elif family == "image":
        image_export.export_image(normalized_payload, target, **export_options)
    elif family == "hdf5":
        hdf5_export.export_hdf5(normalized_payload, target, **export_options)
    else:
        raise ValueError(f"Unsupported export family '{family}'.")

    if write_manifest_file:
        write_manifest(
            normalized_payload,
            target,
            family=family,
            options=export_options,
        )

    return target


__all__ = ["save_selection", "list_supported_formats", "SelectionExportPayload"]
