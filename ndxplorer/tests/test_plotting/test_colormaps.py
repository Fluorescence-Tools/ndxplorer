"""Tests for plotting colormap helpers."""

from __future__ import annotations

import numpy as np
import pytest

from ndxplorer.plotting import colormaps


def test_available_colormaps_include_pyqtgraph_defaults() -> None:
    """Colormap discovery should work without requiring guiqwt."""
    pytest.importorskip("pyqtgraph")

    names = colormaps.get_available_colormaps()

    assert "viridis" in names


def test_create_colormap_lut_returns_uint8_rgba() -> None:
    """LUT generation should use the pyqtgraph-compatible RGBA shape."""
    pytest.importorskip("pyqtgraph")

    lut = colormaps.create_colormap_lut("viridis", n_colors=16)

    assert lut.shape == (16, 4)
    assert lut.dtype == np.uint8


def test_apply_colormap_to_data_returns_rgba_image() -> None:
    """Colorizing a 2D histogram should preserve image dimensions."""
    pytest.importorskip("pyqtgraph")
    data = np.array([[0.0, 0.5], [1.0, 2.0]])

    rgba = colormaps.apply_colormap_to_data(data, "viridis", vmin=0.0, vmax=2.0)

    assert rgba.shape == (2, 2, 4)
    assert rgba.dtype == np.uint8
