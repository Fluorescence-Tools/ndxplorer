"""CSV/TSV export helpers for NDXplorer selections."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from ..logging_config import logging
from ..utils.performance_optimizations import get_performance_monitor
from .models import SelectionExportPayload

_DEFAULT_DELIMITER = ","


def _guess_delimiter(target: Path) -> str:
    suffix = target.suffix.lower()
    if suffix in {".tsv", ".tab"}:
        return "\t"
    if suffix == ".csv":
        return ","
    return _DEFAULT_DELIMITER


def export_table(
    payload: SelectionExportPayload,
    path: Path,
    *,
    delimiter: Optional[str] = None,
    encoding: str = "utf-8",
    include_metadata: bool = True,
    metadata_path: Optional[Path] = None,
) -> None:
    """
    Write the selection data as CSV/TSV.

    Parameters
    ----------
    payload:
        SelectionExportPayload with tabular data.
    path:
        Destination file path.
    delimiter:
        Column separator. If None, inferred from suffix (.csv/.tsv).
    encoding:
        Text encoding for the CSV file.
    include_metadata:
        When True, write a sidecar JSON file describing selections & metadata.
    metadata_path:
        Optional override for the metadata file location.
    """

    if not payload.has_tabular_data():
        raise ValueError("export_table requires payload.table or payload.values.")

    delimiter = delimiter or _guess_delimiter(path)
    df: pd.DataFrame = payload.as_dataframe()

    logging.info(
        "Exporting %d rows x %d cols to %s (delimiter=%r)",
        len(df.index),
        len(df.columns),
        path,
        delimiter,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    perf = get_performance_monitor()
    op_name = f"export_csv[{path.suffix or 'unknown'}]"
    perf.start_timer(op_name)
    try:
        df.to_csv(path, sep=delimiter, index=False, encoding=encoding)
        if include_metadata:
            meta_path = metadata_path or path.with_suffix(path.suffix + ".meta.json")
            _write_metadata(payload, meta_path)
    finally:
        perf.end_timer(op_name)
        perf.log_memory_usage(op_name)


def _write_metadata(payload: SelectionExportPayload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "name": payload.name,
        "selections": payload.serialized_selections(),
        "metadata": payload.metadata,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logging.debug("Export metadata written to %s", path)


__all__ = ["export_table"]
