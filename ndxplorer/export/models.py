"""Data models and helpers for NDXplorer export workflows."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

SelectionLike = Any


class ExportValidationError(ValueError):
    """Raised when an export payload fails validation."""


def _to_plain_value(value: Any) -> Any:
    """Convert numpy/Pandas scalar types to native Python for serialization."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, pd.Series):
        return value.to_list()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="list")
    return value


def summarize_selection(selection: SelectionLike) -> Mapping[str, Any]:
    """Best-effort conversion of a DataSelection object to a serializable dict."""
    if selection is None:
        return {}

    if dataclasses.is_dataclass(selection):
        return dataclasses.asdict(selection)

    if hasattr(selection, "to_dict"):
        try:
            data = selection.to_dict()
            if isinstance(data, Mapping):
                return data
        except Exception:
            pass

    summary: dict[str, Any] = {"type": selection.__class__.__name__}
    for attr in (
        "name",
        "enabled",
        "invert",
        "parameter_idx",
        "parameter_idx1",
        "parameter_idx2",
        "lower",
        "upper",
        "mu",
        "cov",
        "sigma",
        "log_x",
        "log_y",
    ):
        if hasattr(selection, attr):
            summary[attr] = _to_plain_value(getattr(selection, attr))

    # Include any __dict__ entries that look JSON serializable.
    if hasattr(selection, "__dict__"):
        for key, value in selection.__dict__.items():
            if key.startswith("_"):
                continue
            summary.setdefault(key, _to_plain_value(value))
    return summary


@dataclass
class SelectionExportPayload:
    """
    Container passed to export backends.

    Attributes
    ----------
    selections:
        Original selection objects (RectangularDataSelection, Gaussian2DSelection, etc.).
    table:
        PANDAS DataFrame representing the selected data (optional).
    values:
        Raw numpy array version of the selected data (shape (n_features, n_points)).
    columns:
        Column labels corresponding to `values`.
    figure:
        Matplotlib figure handle for image exports.
    image:
        QImage-like object exposing `.save()` (optional).
    pixmap:
        QPixmap-like object exposing `.save()` (optional).
    metadata:
        Extra metadata to persist inside export artifacts.
    name:
        Optional friendly name for the exported selection set.
    """

    selections: Sequence[SelectionLike] = field(default_factory=list)
    table: Optional[pd.DataFrame] = None
    values: Optional[np.ndarray] = None
    columns: Optional[Sequence[str]] = None
    figure: Any = None
    image: Any = None
    pixmap: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    name: Optional[str] = None

    def has_tabular_data(self) -> bool:
        return self.table is not None or self.values is not None

    def as_dataframe(self) -> pd.DataFrame:
        """Return a pandas.DataFrame view of the payload data."""
        if self.table is not None:
            return self.table
        if self.values is None:
            raise ValueError("Selection payload does not include tabular data.")

        data = self.values
        if data.ndim == 2:
            # Values are stored as (n_parameters, n_points); transpose to rows.
            data = data.T
        df = pd.DataFrame(data, columns=list(self.columns) if self.columns else None)
        self.table = df
        return df

    def serialized_selections(self) -> list[Mapping[str, Any]]:
        """Return selections as JSON-friendly dictionaries."""
        return [summarize_selection(sel) for sel in (self.selections or [])]

    def to_json_metadata(self) -> str:
        """Return a JSON string capturing metadata + selections."""
        payload = {
            "name": self.name,
            "metadata": self.metadata,
            "selections": self.serialized_selections(),
        }
        return json.dumps(payload, indent=2, default=_to_plain_value)

    def has_drawable(self) -> bool:
        """Return True when a figure/image/pixmap is available for export."""
        return any(obj is not None for obj in (self.figure, self.image, self.pixmap))


def validate_payload_integrity(payload: SelectionExportPayload) -> None:
    """
    Run basic consistency checks before exporting.

    Raises
    ------
    ExportValidationError
        If the payload contains truncated/invalid data structures.
    """

    errors: list[str] = []

    if payload.table is not None:
        df = payload.table
        if df.empty:
            errors.append("Tabular export requested but the provided DataFrame is empty.")
        if payload.columns:
            missing = [col for col in payload.columns if col not in df.columns]
            if missing:
                errors.append(
                    "Payload columns do not match DataFrame columns "
                    f"(missing: {', '.join(map(str, missing))})."
                )
    elif payload.values is not None:
        values = np.asarray(payload.values)
        if values.ndim != 2:
            errors.append(
                "Payload values must be a 2D array shaped (n_parameters, n_points)."
            )
        elif payload.columns and len(payload.columns) != values.shape[0]:
            errors.append(
                "Number of columns does not match payload.values shape: "
                f"{len(payload.columns)} columns vs {values.shape[0]} parameters."
            )

    if payload.metadata is None:
        payload.metadata = {}
    elif not isinstance(payload.metadata, Mapping):
        errors.append("Payload metadata must be a mapping/dict.")

    if payload.selections:
        bad_selections = [idx for idx, sel in enumerate(payload.selections) if sel is None]
        if bad_selections:
            errors.append(
                "Selection entries at indices "
                + ", ".join(map(str, bad_selections))
                + " are None – please drop corrupted selections."
            )

    if errors:
        raise ExportValidationError(_format_validation_errors(errors))


def _format_validation_errors(errors: Sequence[str]) -> str:
    header = "Export payload failed validation:"
    bullets = "\n".join(f"- {message}" for message in errors)
    return f"{header}\n{bullets}"


__all__ = [
    "SelectionExportPayload",
    "summarize_selection",
    "ExportValidationError",
    "validate_payload_integrity",
    "_to_plain_value",
]
