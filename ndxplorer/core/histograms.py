"""Histogram data classes for NDXplorer core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class Histogram1D:
    """1D histogram container with edges and counts."""
    edges: np.ndarray
    counts: np.ndarray

    @property
    def n_bins(self) -> int:
        """Get number of bins."""
        return len(self.edges) - 1

    def validate(self) -> bool:
        """Validate histogram consistency."""
        try:
            if len(self.counts.shape) != 1:
                return False

            if len(self.edges.shape) != 1:
                return False

            if len(self.counts) != len(self.edges) - 1:
                return False

            if not np.all(np.diff(self.edges) > 0):
                return False

            return True

        except Exception:
            return False

    def as_tuple(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return as (edges, counts) tuple."""
        return self.edges, self.counts


@dataclass
class Histogram2D:
    """2D histogram container with H matrix and axis edges."""
    H: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray

    @property
    def shape(self) -> Tuple[int, int]:
        """Get histogram shape."""
        return self.H.shape

    @property
    def n_bins_x(self) -> int:
        """Get number of X bins."""
        return len(self.x_edges) - 1

    @property
    def n_bins_y(self) -> int:
        """Get number of Y bins."""
        return len(self.y_edges) - 1

    def validate(self) -> bool:
        """Validate histogram consistency."""
        try:
            if len(self.H.shape) != 2:
                return False

            if len(self.x_edges.shape) != 1 or len(self.y_edges.shape) != 1:
                return False

            if self.H.shape[1] != len(self.x_edges) - 1:
                return False

            if self.H.shape[0] != len(self.y_edges) - 1:
                return False

            if not np.all(np.diff(self.x_edges) > 0):
                return False

            if not np.all(np.diff(self.y_edges) > 0):
                return False

            return True

        except Exception:
            return False

    def as_tuple(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return as (H, x_edges, y_edges) tuple."""
        return self.H, self.x_edges, self.y_edges

    def get_x_marginal(self) -> Histogram1D:
        """Get X marginal histogram (sum along Y axis)."""
        counts = np.sum(self.H, axis=0)
        return Histogram1D(edges=self.x_edges, counts=counts)

    def get_y_marginal(self) -> Histogram1D:
        """Get Y marginal histogram (sum along X axis)."""
        counts = np.sum(self.H, axis=1)
        return Histogram1D(edges=self.y_edges, counts=counts)


@dataclass
class Histogram3D:
    """3D histogram container with H tensor and axis edges."""
    H: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray
    z_edges: np.ndarray

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get histogram shape."""
        return self.H.shape

    @property
    def n_bins_x(self) -> int:
        """Get number of X bins."""
        return len(self.x_edges) - 1

    @property
    def n_bins_y(self) -> int:
        """Get number of Y bins."""
        return len(self.y_edges) - 1

    @property
    def n_bins_z(self) -> int:
        """Get number of Z bins."""
        return len(self.z_edges) - 1

    def validate(self) -> bool:
        """Validate histogram consistency."""
        try:
            if len(self.H.shape) != 3:
                return False

            if len(self.x_edges.shape) != 1 or len(self.y_edges.shape) != 1 or len(self.z_edges.shape) != 1:
                return False

            if self.H.shape[2] != len(self.x_edges) - 1:
                return False

            if self.H.shape[1] != len(self.y_edges) - 1:
                return False

            if self.H.shape[0] != len(self.z_edges) - 1:
                return False

            if not np.all(np.diff(self.x_edges) > 0):
                return False

            if not np.all(np.diff(self.y_edges) > 0):
                return False

            if not np.all(np.diff(self.z_edges) > 0):
                return False

            return True

        except Exception:
            return False

    def as_tuple(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return as (H, x_edges, y_edges, z_edges) tuple."""
        return self.H, self.x_edges, self.y_edges, self.z_edges

    def get_xy_marginal(self) -> Histogram2D:
        """Get XY marginal histogram (sum along Z axis)."""
        H = np.sum(self.H, axis=0)
        return Histogram2D(H=H, x_edges=self.x_edges, y_edges=self.y_edges)

    def get_xz_marginal(self) -> Histogram2D:
        """Get XZ marginal histogram (sum along Y axis)."""
        H = np.sum(self.H, axis=1)
        return Histogram2D(H=H, x_edges=self.x_edges, y_edges=self.z_edges)

    def get_yz_marginal(self) -> Histogram2D:
        """Get YZ marginal histogram (sum along X axis)."""
        H = np.sum(self.H, axis=2)
        return Histogram2D(H=H, x_edges=self.y_edges, y_edges=self.z_edges)

    def get_x_marginal(self) -> Histogram1D:
        """Get X marginal histogram (sum along Y and Z axes)."""
        counts = np.sum(self.H, axis=(0, 1))
        return Histogram1D(edges=self.x_edges, counts=counts)

    def get_y_marginal(self) -> Histogram1D:
        """Get Y marginal histogram (sum along X and Z axes)."""
        counts = np.sum(self.H, axis=(0, 2))
        return Histogram1D(edges=self.y_edges, counts=counts)

    def get_z_marginal(self) -> Histogram1D:
        """Get Z marginal histogram (sum along X and Y axes)."""
        counts = np.sum(self.H, axis=(1, 2))
        return Histogram1D(edges=self.z_edges, counts=counts)
