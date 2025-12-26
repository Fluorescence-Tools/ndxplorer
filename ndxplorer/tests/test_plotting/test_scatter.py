"""Unit tests for scatter plot functionality."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from ndxplorer.plotting import scatter


class TestScatter:
    """Test scatter plot functions."""

    def test_create_scatter_plot_basic(self):
        """Test basic scatter plot creation."""
        ndxplorer = Mock()
        x_data = np.array([1, 2, 3, 4])
        y_data = np.array([1, 2, 3, 4])
        
        plot_data = scatter.create_scatter_plot(ndxplorer, x_data, y_data)
        
        assert np.array_equal(plot_data["x"], x_data)
        assert np.array_equal(plot_data["y"], y_data)
        assert plot_data["n_points"] == 4
        assert not plot_data["has_color_mapping"]
        assert not plot_data["has_size_mapping"]
        assert not plot_data["has_weights"]
        assert plot_data["alpha"] == 0.7

    def test_create_scatter_plot_with_color(self):
        """Test scatter plot creation with color mapping."""
        ndxplorer = Mock()
        x_data = np.array([1, 2, 3, 4])
        y_data = np.array([1, 2, 3, 4])
        color_data = np.array([0.1, 0.5, 0.9, 1.0])
        
        plot_data = scatter.create_scatter_plot(ndxplorer, x_data, y_data, color_data=color_data)
        
        assert np.array_equal(plot_data["colors"], color_data)
        assert plot_data["has_color_mapping"]
        assert not plot_data["has_size_mapping"]

    def test_create_scatter_plot_with_size(self):
        """Test scatter plot creation with size mapping."""
        ndxplorer = Mock()
        x_data = np.array([1, 2, 3, 4])
        y_data = np.array([1, 2, 3, 4])
        size_data = np.array([10, 20, 30, 40])
        
        plot_data = scatter.create_scatter_plot(ndxplorer, x_data, y_data, size_data=size_data)
        
        assert np.array_equal(plot_data["sizes"], size_data)
        assert not plot_data["has_color_mapping"]
        assert plot_data["has_size_mapping"]

    def test_create_scatter_plot_with_weights(self):
        """Test scatter plot creation with weights."""
        ndxplorer = Mock()
        x_data = np.array([1, 2, 3, 4])
        y_data = np.array([1, 2, 3, 4])
        weights = np.array([0.5, 0.7, 0.9, 1.0])
        
        plot_data = scatter.create_scatter_plot(ndxplorer, x_data, y_data, weights=weights)
        
        assert plot_data["has_weights"]
        # Alpha should be modified by weights
        assert len(plot_data["alpha"]) == 4

    def test_create_scatter_plot_mismatched_lengths(self):
        """Test scatter plot creation with mismatched array lengths."""
        ndxplorer = Mock()
        x_data = np.array([1, 2, 3])
        y_data = np.array([1, 2, 3, 4])  # Different length
        
        with pytest.raises(ValueError, match="x_data and y_data must have the same length"):
            scatter.create_scatter_plot(ndxplorer, x_data, y_data)

    def test_create_weighted_scatter(self):
        """Test weighted scatter plot creation."""
        ndxplorer = Mock()
        x_data = np.array([1, 2, 3, 4])
        y_data = np.array([1, 2, 3, 4])
        weights = np.array([1, 2, 3, 4])
        
        plot_data = scatter.create_weighted_scatter(
            ndxplorer, x_data, y_data, weights, size_scale=100
        )
        
        assert np.array_equal(plot_data["x"], x_data)
        assert np.array_equal(plot_data["y"], y_data)
        assert plot_data["has_weights"]
        assert plot_data["has_color_mapping"]
        assert plot_data["has_size_mapping"]
        # Sizes should be scaled by weights
        expected_sizes = 10 + (weights / 4) * 100
        assert np.allclose(plot_data["sizes"], expected_sizes)

    def test_apply_selection_to_scatter(self):
        """Test applying selection mask to scatter plot."""
        original_data = {
            "x": np.array([1, 2, 3, 4]),
            "y": np.array([1, 2, 3, 4]),
            "colors": np.array([0.1, 0.2, 0.3, 0.4]),
            "sizes": np.array([10, 20, 30, 40]),
            "alpha": 0.7,
            "n_points": 4,
            "has_color_mapping": True,
            "has_size_mapping": True,
            "has_weights": False
        }
        
        selection_mask = np.array([True, False, True, False])
        
        modified_data = scatter.apply_selection_to_scatter(
            original_data, selection_mask, unselected_alpha=0.2
        )
        
        assert modified_data["n_selected"] == 2
        assert np.array_equal(modified_data["selection"], selection_mask)
        # Selected points should have original alpha, unselected should have reduced alpha
        assert modified_data["alpha_array"][0] == 0.7  # Selected
        assert modified_data["alpha_array"][1] == 0.2  # Unselected
        assert modified_data["alpha_array"][2] == 0.7  # Selected
        assert modified_data["alpha_array"][3] == 0.2  # Unselected

    def test_compute_scatter_statistics_basic(self):
        """Test basic scatter statistics computation."""
        ndxplorer = Mock()
        x_data = np.array([1, 2, 3, 4, 5])
        y_data = np.array([2, 4, 6, 8, 10])
        
        stats = scatter.compute_scatter_statistics(ndxplorer, x_data, y_data)
        
        assert stats["n_points"] == 5
        assert stats["x_mean"] == 3.0
        assert stats["y_mean"] == 6.0
        assert stats["x_range"] == (1.0, 5.0)
        assert stats["y_range"] == (2.0, 10.0)
        assert "correlation" in stats

    def test_compute_scatter_statistics_with_weights(self):
        """Test scatter statistics computation with weights."""
        ndxplorer = Mock()
        x_data = np.array([1, 2, 3, 4])
        y_data = np.array([2, 4, 6, 8])
        weights = np.array([1, 2, 1, 2])
        
        stats = scatter.compute_scatter_statistics(ndxplorer, x_data, y_data, weights)
        
        assert stats["total_weight"] == 6.0
        assert "weighted_x_mean" in stats
        assert "weighted_y_mean" in stats

    @patch('ndxplorer.plotting.scatter.gaussian_kde')
    def test_create_density_scatter(self, mock_kde):
        """Test density scatter plot creation."""
        # Mock KDE
        mock_kde.return_value.return_value = np.array([0.1, 0.2, 0.3, 0.4])
        
        ndxplorer = Mock()
        x_data = np.array([1, 2, 3, 4])
        y_data = np.array([1, 2, 3, 4])
        
        plot_data = scatter.create_density_scatter(ndxplorer, x_data, y_data)
        
        assert np.array_equal(plot_data["x"], x_data)
        assert np.array_equal(plot_data["y"], y_data)
        assert plot_data["has_color_mapping"]
        assert plot_data["has_size_mapping"]

    @patch('ndxplorer.plotting.scatter.gaussian_kde')
    def test_create_density_scatter_fallback(self, mock_kde):
        """Test density scatter plot fallback when scipy unavailable."""
        mock_kde.side_effect = ImportError("scipy not available")
        
        ndxplorer = Mock()
        x_data = np.array([1, 2, 3, 4])
        y_data = np.array([1, 2, 3, 4])
        
        plot_data = scatter.create_density_scatter(ndxplorer, x_data, y_data)
        
        # Should fall back to regular scatter
        assert not plot_data["has_color_mapping"]
        assert not plot_data["has_size_mapping"]

    def test_export_scatter_data_csv(self):
        """Test exporting scatter data to CSV."""
        plot_data = {
            "x": np.array([1, 2, 3]),
            "y": np.array([4, 5, 6]),
            "colors": np.array([0.1, 0.2, 0.3]),
            "sizes": np.array([10, 20, 30])
        }
        
        csv_output = scatter.export_scatter_data(plot_data, format="csv")
        
        assert "x,y,colors,sizes" in csv_output
        assert "1,4,0.1,10.0" in csv_output

    def test_export_scatter_data_json(self):
        """Test exporting scatter data to JSON."""
        plot_data = {
            "x": np.array([1, 2, 3]),
            "y": np.array([4, 5, 6]),
            "colors": np.array([0.1, 0.2, 0.3]),
            "sizes": np.array([10, 20, 30])
        }
        
        json_output = scatter.export_scatter_data(plot_data, format="json")
        
        import json
        data = json.loads(json_output)
        assert len(data) == 3
        assert data[0]["x"] == 1.0
        assert data[0]["y"] == 4.0

    def test_export_scatter_data_numpy(self):
        """Test exporting scatter data to numpy format."""
        plot_data = {
            "x": np.array([1, 2, 3]),
            "y": np.array([4, 5, 6]),
            "colors": np.array([0.1, 0.2, 0.3]),
            "sizes": np.array([10, 20, 30])
        }
        
        output = scatter.export_scatter_data(plot_data, format="numpy")
        
        assert isinstance(output, dict)
        assert np.array_equal(output["x"], np.array([1, 2, 3]))
        assert np.array_equal(output["y"], np.array([4, 5, 6]))

    def test_export_scatter_data_unsupported_format(self):
        """Test exporting scatter data with unsupported format."""
        plot_data = {"x": np.array([1, 2, 3])}
        
        with pytest.raises(ValueError, match="Unsupported format"):
            scatter.export_scatter_data(plot_data, format="unsupported")


if __name__ == "__main__":
    pytest.main([__file__])
