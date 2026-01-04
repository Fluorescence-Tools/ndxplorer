"""End-to-end workflow tests covering plotting/export integration paths."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ndxplorer.export.api import save_selection
from ndxplorer.tests.fixtures.dataset_fixtures import (
    create_synthetic_selection_payload,
    write_fixture_bundle,
)


@pytest.mark.integration
def test_export_round_trip_with_synthetic_payload(tmp_path):
    """Workflow: synthetic payload -> CSV/HDF5 export -> data parity checks."""
    payload = create_synthetic_selection_payload(
        n_points=10_000,
        seed=314,
        include_figure=False,
    )

    csv_path = tmp_path / "synthetic.csv"
    hdf5_path = tmp_path / "synthetic.h5"

    save_selection(payload, csv_path, format="csv")
    save_selection(payload, hdf5_path, format="hdf5")

    csv_df = pd.read_csv(csv_path)
    hdf5_df = pd.read_hdf(hdf5_path)

    expected_rows = payload.values.shape[1]
    assert len(csv_df) == expected_rows
    assert len(hdf5_df) == expected_rows
    assert list(csv_df.columns) == list(hdf5_df.columns)
    assert list(csv_df.columns) == list(payload.columns)

    meta_path = csv_path.with_suffix(csv_path.suffix + ".meta.json")
    assert meta_path.exists()
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    assert metadata["name"] == payload.name
    assert metadata["metadata"]["n_points"] == expected_rows


@pytest.mark.integration
def test_fixture_bundle_produces_expected_artifacts(tmp_path):
    """Workflow: dataset fixture writer generates all artifact types."""
    summary = write_fixture_bundle(tmp_path, n_points=5_000, seed=2718)
    paths = {key: Path(value) for key, value in summary.items() if value}

    required_keys = {"csv", "hdf5", "npz", "metadata"}
    assert required_keys.issubset(paths.keys())

    for path in paths.values():
        assert path.exists(), f"Missing fixture artifact: {path}"

    corrupted_manifest = paths.get("corrupted_manifest")
    assert corrupted_manifest is not None
    info = json.loads(corrupted_manifest.read_text(encoding="utf-8"))
    assert info["description"].startswith("Corrupted payload")
    assert info["values_shape"][1] == 5_000
