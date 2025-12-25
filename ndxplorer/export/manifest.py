"""Helpers for writing redundant export manifest files."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Any
import json
from datetime import datetime, timezone

from .models import SelectionExportPayload, _to_plain_value


def _default_serializer(value: Any) -> Any:
    return _to_plain_value(value)


def _build_data_summary(payload: SelectionExportPayload) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "has_tabular_data": payload.has_tabular_data(),
        "has_drawable": payload.has_drawable(),
    }

    if payload.has_tabular_data():
        df = payload.as_dataframe()
        summary.update(
            {
                "rows": len(df.index),
                "columns": list(df.columns),
                "column_count": len(df.columns),
            }
        )
    elif payload.values is not None:
        summary["values_shape"] = list(payload.values.shape)

    summary["metadata_keys"] = list((payload.metadata or {}).keys())
    summary["selection_count"] = len(payload.selections or [])
    return summary


def write_manifest(
    payload: SelectionExportPayload,
    target: Path,
    *,
    family: str,
    options: Mapping[str, Any],
) -> Path:
    """
    Write a manifest JSON file next to the export artifact.
    """

    manifest_path = target.with_suffix(target.suffix + ".manifest.json")
    manifest = {
        "file": str(target),
        "family": family,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "options": dict(options),
        "payload": {
            "name": payload.name,
            "metadata": payload.metadata,
            "selections": payload.serialized_selections(),
        },
        "data_summary": _build_data_summary(payload),
    }

    try:
        manifest["file_size_bytes"] = target.stat().st_size
    except OSError:
        manifest["file_size_bytes"] = None

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=_default_serializer),
        encoding="utf-8",
    )
    return manifest_path


__all__ = ["write_manifest"]
