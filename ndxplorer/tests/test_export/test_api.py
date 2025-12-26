"""
Tests for export API facade.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ndxplorer.export.api import (
    save_selection,
    SelectionExportPayload,
    ExportValidationError,
)


class TestExportAPI:
    """Test unified export API."""

    @pytest.fixture
    def sample_payload(self):
        """Create sample SelectionExportPayload for testing."""
        df = pd.DataFrame({
            'x': np.arange(10),
            'y': np.random.random(10)
        })
        return SelectionExportPayload(table=df, name="test-selection")
    
    def test_csv_export_via_api(self, sample_payload):
        """Test CSV export through API."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.csv"
            save_selection(sample_payload, filepath, format='csv')

            assert filepath.exists()
            assert filepath.suffix == '.csv'
    
    def test_png_export_via_api(self, sample_payload):
        """Test PNG export through API."""
        fig = _build_matplotlib_figure()
        payload = SelectionExportPayload(
            table=sample_payload.table,
            figure=fig,
            name="figure-selection"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.png"
            save_selection(payload, filepath, format='image')

            assert filepath.exists()
            assert filepath.suffix == '.png'
    
    def test_hdf5_export_via_api(self, sample_payload):
        """Test HDF5 export through API."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.h5"
            save_selection(sample_payload, filepath, format='hdf5')

            assert filepath.exists()
            assert filepath.suffix == '.h5'
    
    def test_auto_format_detection(self, sample_payload):
        """Test automatic format detection from file extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            save_selection(sample_payload, csv_path)
            assert csv_path.exists()

            png_path = Path(tmpdir) / "test.png"
            payload = SelectionExportPayload(
                table=sample_payload.table,
                figure=_build_matplotlib_figure(),
                name="figure-selection"
            )
            save_selection(payload, png_path)
            assert png_path.exists()

            hdf5_path = Path(tmpdir) / "test.h5"
            save_selection(sample_payload, hdf5_path)
            assert hdf5_path.exists()
    
    def test_invalid_format_handling(self, sample_payload):
        """Test handling of invalid format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.invalid"

            with pytest.raises(ValueError):
                save_selection(sample_payload, filepath, format='invalid')
    
    def test_export_with_options(self, sample_payload):
        """Test export with format-specific options."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.csv"
            save_selection(sample_payload, filepath, format='csv',
                           options={'delimiter': '|'})

            assert filepath.exists()
            with open(filepath, 'r', encoding="utf-8") as f:
                content = f.read()
                assert '|' in content

    def test_payload_validation_errors(self):
        """Ensure save_selection raises ExportValidationError for bad payloads."""
        payload = SelectionExportPayload(table=pd.DataFrame())
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "bad.csv"
            with pytest.raises(ExportValidationError) as excinfo:
                save_selection(payload, filepath, format='csv')
        message = str(excinfo.value)
        assert "Export payload failed validation" in message
        assert "- Tabular export requested but the provided DataFrame is empty." in message

    def test_image_export_requires_drawable(self, sample_payload):
        """Ensure image export fails without drawable data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.png"
            with pytest.raises(ExportValidationError):
                save_selection(sample_payload, filepath, format='image')

    def test_payload_values_column_mismatch(self):
        """Validator should flag column/value shape mismatches."""
        values = np.zeros((2, 5))
        payload = SelectionExportPayload(values=values, columns=["x", "y", "z"])
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "bad.csv"
            with pytest.raises(ExportValidationError) as excinfo:
                save_selection(payload, filepath, format='csv')
        assert "Number of columns does not match payload.values shape" in str(excinfo.value)

    def test_payload_metadata_must_be_mapping(self):
        """Validator should enforce mapping metadata."""
        payload = SelectionExportPayload(
            table=pd.DataFrame({"x": [1]}),
            metadata=["not", "mapping"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "bad.csv"
            with pytest.raises(ExportValidationError) as excinfo:
                save_selection(payload, filepath, format='csv')
        assert "Payload metadata must be a mapping" in str(excinfo.value)

    def test_payload_none_selection_entries(self):
        """Validator should report None entries in selections."""
        payload = SelectionExportPayload(
            table=pd.DataFrame({"x": [1]}),
            selections=[None, object()],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "bad.csv"
            with pytest.raises(ExportValidationError) as excinfo:
                save_selection(payload, filepath, format='csv')
        assert "Selection entries at indices 0" in str(excinfo.value)

    def test_manifest_written_by_default(self, sample_payload):
        """Manifest sidecar should be written automatically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.csv"
            save_selection(sample_payload, filepath, format='csv')

            manifest_path = filepath.with_suffix(filepath.suffix + ".manifest.json")
            assert manifest_path.exists()
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            assert data["family"] == "csv"
            assert data["data_summary"]["rows"] == len(sample_payload.table.index)

    def test_manifest_can_be_disabled(self, sample_payload):
        """write_manifest_file flag should skip manifest creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.csv"
            save_selection(
                sample_payload,
                filepath,
                format='csv',
                write_manifest_file=False,
            )

            manifest_path = filepath.with_suffix(filepath.suffix + ".manifest.json")
            assert not manifest_path.exists()

    def test_payload_metadata_must_be_mapping(self):
        """metadata must be dict-like."""
        payload = SelectionExportPayload(
            values=np.ones((2, 2)),
            metadata=["not", "a", "mapping"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "bad.csv"
            with pytest.raises(ExportValidationError):
                save_selection(payload, filepath, format='csv')

    def test_payload_none_selection_entries(self):
        """None selections should trigger validation error."""
        payload = SelectionExportPayload(
            table=pd.DataFrame({"x": [1, 2]}),
            selections=[None, object()],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "bad.csv"
            with pytest.raises(ExportValidationError):
                save_selection(payload, filepath, format='csv')


def _build_matplotlib_figure():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_title("Test Figure")
    return fig
