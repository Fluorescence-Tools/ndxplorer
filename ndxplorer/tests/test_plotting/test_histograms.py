"""Unit tests for histogram plotting functionality."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from ndxplorer.plotting import histograms


class TestHistograms:
    """Test histogram plotting functions."""

    def test_plot_histogram_2d(self):
        """Test 2D histogram plotting."""
        # Mock NDXplorer instance
        ndxplorer = Mock()
        ndxplorer._histogram = {
            "2d": (np.array([[1, 2], [3, 4]]), np.array([0, 1, 2]), np.array([0, 1, 2]))
        }
        
        H, (x_edges, y_edges) = histograms.plot_histogram(ndxplorer, "2d")
        
        assert np.array_equal(H, np.array([[1, 2], [3, 4]]))
        assert np.array_equal(x_edges, np.array([0, 1, 2]))
        assert np.array_equal(y_edges, np.array([0, 1, 2]))

    def test_plot_histogram_1d(self):
        """Test 1D histogram plotting."""
        ndxplorer = Mock()
        ndxplorer._histogram = {
            "x": (np.array([0, 1, 2, 3]), np.array([5, 10, 15]))
        }
        
        bin_edges, counts = histograms.plot_histogram(ndxplorer, "x")
        
        assert np.array_equal(bin_edges, np.array([0, 1, 2, 3]))
        assert np.array_equal(counts, np.array([5, 10, 15]))

    def test_plot_histogram_invalid_dimension(self):
        """Test histogram plotting with invalid dimension."""
        ndxplorer = Mock()
        ndxplorer._histogram = {}
        
        bin_edges, counts = histograms.plot_histogram(ndxplorer, "invalid")
        
        assert np.array_equal(bin_edges, np.array([0, 1]))
        assert np.array_equal(counts, np.array([0]))

    def test_compute_2d_histogram_basic(self):
        """Test basic 2D histogram computation."""
        ndxplorer = Mock()
        x_data = np.array([1, 2, 3, 4])
        y_data = np.array([1, 2, 3, 4])
        
        H, x_edges, y_edges = histograms.compute_2d_histogram(
            ndxplorer, x_data, y_data, 3, 3
        )
        
        assert H.shape == (2, 2)  # 3 bins -> 2 edges
        assert len(x_edges) == 3
        assert len(y_edges) == 3

    def test_compute_2d_histogram_with_weights(self):
        """Test 2D histogram computation with weights."""
        ndxplorer = Mock()
        x_data = np.array([1, 2, 3, 4])
        y_data = np.array([1, 2, 3, 4])
        weights = np.array([1, 2, 1, 2])
        
        H, x_edges, y_edges = histograms.compute_2d_histogram(
            ndxplorer, x_data, y_data, 3, 3, weights=weights
        )
        
        assert H.shape == (2, 2)
        assert np.sum(H) == np.sum(weights)

    def test_compute_1d_histogram_basic(self):
        """Test basic 1D histogram computation."""
        ndxplorer = Mock()
        data = np.array([1, 2, 3, 4, 5])
        
        bin_edges, counts = histograms.compute_1d_histogram(
            ndxplorer, data, 5
        )
        
        assert len(counts) == 5
        assert len(bin_edges) == 6
        assert np.sum(counts) == len(data)

    def test_compute_1d_histogram_with_weights(self):
        """Test 1D histogram computation with weights."""
        ndxplorer = Mock()
        data = np.array([1, 2, 3, 4, 5])
        weights = np.array([1, 2, 1, 2, 1])
        
        bin_edges, counts = histograms.compute_1d_histogram(
            ndxplorer, data, 5, weights=weights
        )
        
        assert len(counts) == 5
        assert np.sum(counts) == np.sum(weights)

    def test_get_histogram_statistics_2d(self):
        """Test statistics computation for 2D histogram."""
        ndxplorer = Mock()
        ndxplorer._histogram = {
            "2d": (np.array([[1, 2], [3, 4]]), np.array([0, 1, 2]), np.array([0, 1, 2]))
        }
        
        stats = histograms.get_histogram_statistics(ndxplorer, "2d")
        
        assert stats["count"] == 10
        assert stats["mean"] == 2.5
        assert stats["shape"] == (2, 2)
        assert "min" in stats
        assert "max" in stats

    def test_get_histogram_statistics_1d(self):
        """Test statistics computation for 1D histogram."""
        ndxplorer = Mock()
        ndxplorer._histogram = {
            "x": (np.array([5, 10, 15]), np.array([0, 1, 2, 3]))
        }
        
        stats = histograms.get_histogram_statistics(ndxplorer, "x")
        
        assert stats["count"] == 30
        assert stats["mean"] == 10.0
        assert stats["bins"] == 3

    def test_get_histogram_statistics_invalid(self):
        """Test statistics computation with invalid data."""
        ndxplorer = Mock()
        ndxplorer._histogram = {}
        
        stats = histograms.get_histogram_statistics(ndxplorer, "invalid")
        
        assert stats == {}

    @patch('ndxplorer.plotting.histograms.is_data_ready')
    @patch('ndxplorer.plotting.histograms.update_histograms')
    def test_update_histogram_display_ready(self, mock_update, mock_ready):
        """Test histogram display update when data is ready."""
        mock_ready.return_value = True
        
        ndxplorer = Mock()
        histograms.update_histogram_display(ndxplorer)
        
        mock_update.assert_called_once_with(ndxplorer)

    @patch('ndxplorer.plotting.histograms.is_data_ready')
    def test_update_histogram_display_not_ready(self, mock_ready):
        """Test histogram display update when data is not ready."""
        mock_ready.return_value = False
        
        ndxplorer = Mock()
        histograms.update_histogram_display(ndxplorer)
        
        # Should not attempt to update
        assert not hasattr(ndxplorer, 'update_called')


if __name__ == "__main__":
    pytest.main([__file__])
