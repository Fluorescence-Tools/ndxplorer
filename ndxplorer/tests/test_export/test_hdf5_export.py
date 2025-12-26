"""
Tests for HDF5 export functionality (pandas-only backend).
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from ndxplorer.export.hdf5_export import HDFBackendUnavailable, export_hdf5
from ndxplorer.export.models import SelectionExportPayload


class TestHDF5Export:
    """Test pandas-backed HDF5 export."""

    @pytest.fixture
    def payload(self) -> SelectionExportPayload:
        df = pd.DataFrame(
            {
                "x": np.arange(10, dtype=float),
                "y": np.random.random(10),
                "category": ["A", "B"] * 5,
            }
        )
        return SelectionExportPayload(
            selections=[{"name": "test"}],
            table=df,
            metadata={"experiment": "test"},
            name="sample",
        )

    def _read_back(self, path: Path, key: str = "ndxplorer/table") -> pd.DataFrame:
        return pd.read_hdf(path, key=key)

    def test_basic_export(self, payload: SelectionExportPayload) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "basic.h5"
            export_hdf5(payload, path)

            assert path.exists()
            df = self._read_back(path)
            pd.testing.assert_frame_equal(df, payload.table)

    def test_metadata_persisted(self, payload: SelectionExportPayload) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "meta.h5"
            export_hdf5(payload, path)

            with pd.HDFStore(path, "r") as store:
                storer = store.get_storer("ndxplorer/table")
                assert storer.attrs.metadata_json
                assert "test" in storer.attrs.metadata_json

    def test_values_shape_attribute(self) -> None:
        values = np.random.random((3, 20))
        payload = SelectionExportPayload(values=values, columns=["a", "b", "c"])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "values.h5"
            export_hdf5(payload, path)

            with pd.HDFStore(path, "r") as store:
                attrs = store.get_storer("ndxplorer/table").attrs
                assert tuple(attrs.values_shape) == values.shape

    def test_compression_options(self, payload: SelectionExportPayload) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "compressed.h5"
            export_hdf5(payload, path, compression="blosc", compression_level=5)
            assert path.exists()

    def test_missing_tabular_data_raises(self) -> None:
        payload = SelectionExportPayload()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "invalid.h5"
            with pytest.raises(ValueError):
                export_hdf5(payload, path)

    def test_backend_error_wrapped(self, monkeypatch, payload: SelectionExportPayload) -> None:
        def _broken_store(*args, **kwargs):
            raise ImportError("missing pytables")

        monkeypatch.setattr(pd, "HDFStore", _broken_store)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "broken.h5"
            with pytest.raises(HDFBackendUnavailable):
                export_hdf5(payload, path)
