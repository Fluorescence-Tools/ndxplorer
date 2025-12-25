"""HDF5 export helpers for NDXplorer selections."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from ..logging_config import logging
from ..utils.performance_optimizations import get_performance_monitor
from .models import SelectionExportPayload


class HDFBackendUnavailable(RuntimeError):
    """Raised when pandas cannot persist HDF5 (typically missing PyTables)."""


def export_hdf5(
    payload: SelectionExportPayload,
    path: Path,
    *,
    compression: str = "zlib",
    compression_level: int = 4,
    key: str = "ndxplorer/table",
) -> None:
    """
    Persist selection data + metadata in an HDF5 container using pandas only.

    Parameters
    ----------
    payload:
        SelectionExportPayload containing arrays and/or DataFrame.
    path:
        Destination file.
    compression:
        Compression algorithm handled by pandas/HDFStore (e.g. zlib, blosc).
    compression_level:
        Compression level (0-9) where supported by the underlying writer.
    key:
        HDF5 dataset path within the store.
    """

    if not payload.has_tabular_data():
        raise ValueError("export_hdf5 requires payload.table or payload.values.")

    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = payload.as_dataframe()

    store_kwargs = {
        "mode": "w",
        "complevel": compression_level,
        "complib": compression,
    }

    logging.info("Exporting selection to %s via pandas.HDFStore", path)

    perf = get_performance_monitor()
    op_name = f"export_hdf5[{key}]"
    perf.start_timer(op_name)
    try:
        with pd.HDFStore(path, **store_kwargs) as store:
            store.put(key, dataframe, format="table", data_columns=True)
            storer = store.get_storer(key)
            if payload.values is not None:
                storer.attrs.values_shape = np.asarray(payload.values).shape
            storer.attrs.metadata_json = payload.to_json_metadata()
    except (ImportError, ValueError) as exc:  # pragma: no cover - depends on env
        raise HDFBackendUnavailable(
            "pandas could not open an HDF5 store. Install pandas with HDF5 support "
            "(e.g. `pip install pandas[pytables]`)."
        ) from exc
    finally:
        perf.end_timer(op_name)
        perf.log_memory_usage(op_name)


__all__ = ["export_hdf5", "HDFBackendUnavailable"]
