#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for ndXplorer headless CLI commands."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from ndxplorer.cli import filter_cmd, image_cmd


@pytest.fixture
def dummy_burst_dir(tmp_path):
    """Create a dummy burst directory containing .bur files."""
    bur_dir = tmp_path / "bur_output"
    bur_subdir = bur_dir / "bi4_bur"
    bur_subdir.mkdir(parents=True, exist_ok=True)
    
    # Write a dummy .bur file
    dummy_bur = bur_subdir / "measurement_1.bur"
    dummy_bur.write_text(
        "First Photon\tLast Photon\tFirst File\tLast File\tproximity_ratio\tn_photons\tMean Macro Time (ms)\n"
        "100\t200\tfile1.ptu\tfile1.ptu\t0.5\t100\t1000\n"
        "300\t400\tfile1.ptu\tfile1.ptu\t0.2\t40\t2000\n"
        "500\t600\tfile1.ptu\tfile1.ptu\t0.8\t150\t3000\n"
        "700\t800\tfile1.ptu\tfile1.ptu\t0.4\t60\t4000\n",
        encoding="utf-8",
    )
    return bur_dir


def test_cli_filter_subcommand(dummy_burst_dir, tmp_path):
    """Test click command filter_cmd."""
    runner = CliRunner()
    out_dir = tmp_path / "filtered_output"
    
    # Filter with proximity ratio between 0.3 and 0.6
    result = runner.invoke(filter_cmd, [
        "--folder", str(dummy_burst_dir),
        "--select", "proximity_ratio:0.3-0.6",
        "--out", str(out_dir),
        "--skip-nth-row", "1",
    ])
    
    assert result.exit_code == 0
    # Parse json stdout
    res_data = json.loads(result.output)
    assert res_data["n_in"] == 4
    assert res_data["n_out"] == 2 # 0.5 and 0.4
    
    # Check that a .bst file was written
    bst_file = out_dir / "file1.ptu.bst"
    assert bst_file.exists()
    
    # Content of .bst file should contain photon indices for first and last photons
    bst_data = np.loadtxt(bst_file, delimiter="\t")
    # First row is 100 to 200, second is 700 to 800
    assert np.allclose(bst_data, [[100, 200], [700, 800]])


def test_cli_filter_with_query(dummy_burst_dir, tmp_path):
    """Test click filter_cmd with pandas query."""
    runner = CliRunner()
    out_dir = tmp_path / "filtered_output_query"
    
    # Filter with query: n_photons > 50
    result = runner.invoke(filter_cmd, [
        "--folder", str(dummy_burst_dir),
        "--query", "n_photons > 50",
        "--out", str(out_dir),
        "--skip-nth-row", "1",
    ])
    
    assert result.exit_code == 0
    res_data = json.loads(result.output)
    assert res_data["n_out"] == 3 # 100, 150, 60
    
    bst_file = out_dir / "file1.ptu.bst"
    assert bst_file.exists()


@pytest.fixture
def dummy_image_hdf5(tmp_path):
    """Create a dummy HDF5 with image-axis data."""
    h5_path = tmp_path / "image_data.h5"
    # Create simple columns: x pixel, y pixel, intensity, lifetime
    df = pd.DataFrame({
        "x pixel": [0, 0, 1, 1, 0, 1],
        "y pixel": [0, 1, 0, 1, 0, 1],
        "intensity": [10, 20, 30, 40, 50, 60],
        "lifetime": [1.5, 2.5, 3.5, 4.5, 1.0, 4.0],
    })
    df.to_hdf(h5_path, key="data")
    return h5_path


def test_cli_image_subcommand(dummy_image_hdf5, tmp_path):
    """Test click command image_cmd."""
    runner = CliRunner()
    out_img = tmp_path / "intensity_map.png"

    result = runner.invoke(image_cmd, [
        "--file", str(dummy_image_hdf5),
        "--map", "intensity",
        "--out", str(out_img),
    ])

    assert result.exit_code == 0
    res_data = json.loads(result.output)
    assert res_data["map"] == "intensity"
    assert res_data["shape"] == [2, 2] # 0 to 1 max + 1
    assert out_img.exists()


def test_cli_image_lifetime_map_tiff(dummy_image_hdf5, tmp_path):
    """Lifetime map exports a quantitative (float) TIFF of per-pixel means."""
    from PIL import Image

    runner = CliRunner()
    out_map = tmp_path / "lifetime_map.tiff"

    result = runner.invoke(image_cmd, [
        "--file", str(dummy_image_hdf5),
        "--map", "lifetime",
        "--out", str(out_map),
    ])

    assert result.exit_code == 0, result.output
    res_data = json.loads(result.output)
    assert res_data["map"] == "lifetime"
    assert res_data["shape"] == [2, 2]
    assert out_map.exists()

    # TIFF is the quantitative float map; per-pixel means:
    #   (0,0): mean(1.5, 1.0)=1.25  (0,1): 2.5  (1,0): 3.5  (1,1): mean(4.5, 4.0)=4.25
    arr = np.array(Image.open(out_map)).astype(float)
    assert arr.dtype == float
    nz = np.sort(np.unique(arr[arr > 0]))
    assert np.allclose(nz, [1.25, 2.5, 3.5, 4.25])


def test_cli_image_roi_masks_render(dummy_image_hdf5, tmp_path):
    """An ROI mask excluding one pixel reduces the rendered pixels (DoD: --roi masks it)."""
    from PIL import Image

    # 2x2 ROI mask selecting 3 of the 4 pixels (exclude one corner).
    roi_path = tmp_path / "roi.tiff"
    mask = np.array([[255, 255], [255, 0]], dtype=np.uint8)
    Image.fromarray(mask).save(roi_path)

    runner = CliRunner()
    out_img = tmp_path / "roi_intensity.tiff"
    result = runner.invoke(image_cmd, [
        "--file", str(dummy_image_hdf5),
        "--map", "intensity",
        "--roi", str(roi_path),
        "--out", str(out_img),
    ])

    assert result.exit_code == 0, result.output
    res_data = json.loads(result.output)
    # n_selected_px is the count of ROI-selected pixels (3 of 4).
    assert res_data["n_selected_px"] == 3

    # All 4 pixels are occupied without an ROI; the mask drops exactly one.
    arr = np.array(Image.open(out_img)).astype(float)
    assert int(np.count_nonzero(arr)) == 3
