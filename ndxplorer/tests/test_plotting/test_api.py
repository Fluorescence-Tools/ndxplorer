"""Unit tests for unified plotting API."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from ndxplorer.plotting import api


class TestPlottingAPI:
    """Test unified plotting API functions."""

    def test_plot_histogram_2d(self):
        """Test histogram plotting through API."""
        ndxplorer = Mock()
        ndxplorer._histogram = {
            "2d": (np.array([[1, 2], [3, 4]]), np.array([0, 1, 2]), np.array([0, 1, 2]))
        }
        
        with patch('ndxplorer.plotting.histograms.plot_histogram') as mock_hist:
            mock_hist.return_value = (np.array([[1, 2], [3, 4]]), (np.array([0, 1, 2]), np.array([0, 1, 2])))
            
            result = api.plot_histogram(ndxplorer, "2d")
            
            mock_hist.assert_called_once_with(ndxplorer, "2d")
            assert result is not None

    def test_plot_histogram_1d(self):
        """Test 1D histogram plotting through API."""
        ndxplorer = Mock()
        ndxplorer._histogram = {
            "x": (np.array([5, 10, 15]), np.array([0, 1, 2, 3]))
        }
        
        with patch('ndxplorer.plotting.histograms.plot_histogram') as mock_hist:
            mock_hist.return_value = (np.array([5, 10, 15]), np.array([0, 1, 2, 3]))
            
            result = api.plot_histogram(ndxplorer, "x")
            
            mock_hist.assert_called_once_with(ndxplorer, "x")
            assert result is not None

    def test_plot_scatter_standard(self):
        """Test standard scatter plot through API."""
        ndxplorer = Mock()
        ndxplorer.x_values = np.array([1, 2, 3, 4])
        ndxplorer.y_values = np.array([1, 2, 3, 4])
        
        with patch('ndxplorer.plotting.scatter.create_scatter_plot') as mock_scatter:
            mock_scatter.return_value = {"x": ndxplorer.x_values, "y": ndxplorer.y_values}
            
            result = api.plot_scatter(ndxplorer, plot_type="standard")
            
            mock_scatter.assert_called_once_with(
                ndxplorer, ndxplorer.x_values, ndxplorer.y_values, None, None, None
            )
            assert result is not None

    def test_plot_scatter_with_custom_data(self):
        """Test scatter plot with custom data through API."""
        ndxplorer = Mock()
        x_data = np.array([1, 2, 3])
        y_data = np.array([4, 5, 6])
        color_data = np.array([0.1, 0.2, 0.3])
        
        with patch('ndxplorer.plotting.scatter.create_scatter_plot') as mock_scatter:
            mock_scatter.return_value = {"x": x_data, "y": y_data, "colors": color_data}
            
            result = api.plot_scatter(
                ndxplorer, x_data=x_data, y_data=y_data, color_data=color_data
            )
            
            mock_scatter.assert_called_once_with(
                ndxplorer, x_data, y_data, color_data, None, None
            )
            assert result is not None

    def test_plot_scatter_weighted(self):
        """Test weighted scatter plot through API."""
        ndxplorer = Mock()
        ndxplorer.x_values = np.array([1, 2, 3, 4])
        ndxplorer.y_values = np.array([1, 2, 3, 4])
        weights = np.array([1, 2, 3, 4])
        
        with patch('ndxplorer.plotting.scatter.create_weighted_scatter') as mock_scatter:
            mock_scatter.return_value = {"x": ndxplorer.x_values, "y": ndxplorer.y_values}
            
            result = api.plot_scatter(ndxplorer, weights=weights, plot_type="weighted")
            
            mock_scatter.assert_called_once_with(
                ndxplorer, ndxplorer.x_values, ndxplorer.y_values, weights
            )
            assert result is not None

    def test_plot_scatter_density(self):
        """Test density scatter plot through API."""
        ndxplorer = Mock()
        ndxplorer.x_values = np.array([1, 2, 3, 4])
        ndxplorer.y_values = np.array([1, 2, 3, 4])
        
        with patch('ndxplorer.plotting.scatter.create_density_scatter') as mock_scatter:
            mock_scatter.return_value = {"x": ndxplorer.x_values, "y": ndxplorer.y_values}
            
            result = api.plot_scatter(ndxplorer, plot_type="density")
            
            mock_scatter.assert_called_once_with(ndxplorer, ndxplorer.x_values, ndxplorer.y_values)
            assert result is not None

    def test_apply_colormap_with_data(self):
        """Test colormap application with custom data through API."""
        ndxplorer = Mock()
        data = np.array([[1, 2], [3, 4]])
        
        with patch('ndxplorer.plotting.colormaps.apply_colormap_to_data') as mock_cmap:
            mock_cmap.return_value = np.array([[[255, 0, 0, 255], [0, 255, 0, 255]]])
            
            result = api.apply_colormap(ndxplorer, data=data, colormap_name="viridis")
            
            mock_cmap.assert_called_once_with(data, "viridis", None, None)
            assert result is not None

    def test_apply_colormap_auto_data(self):
        """Test colormap application with auto data through API."""
        ndxplorer = Mock()
        ndxplorer._histogram = {
            "2d": (np.array([[1, 2], [3, 4]]), np.array([0, 1, 2]), np.array([0, 1, 2]))
        }
        ndxplorer.comboBoxCmap = Mock()
        ndxplorer.comboBoxCmap.currentText.return_value = "plasma"
        
        with patch('ndxplorer.plotting.colormaps.apply_colormap_to_data') as mock_cmap:
            mock_cmap.return_value = np.array([[[255, 0, 0, 255], [0, 255, 0, 255]]])
            
            result = api.apply_colormap(ndxplorer)
            
            mock_cmap.assert_called_once()
            assert result is not None

    def test_apply_colormap_no_data(self):
        """Test colormap application with no data available through API."""
        ndxplorer = Mock()
        ndxplorer._histogram = {}
        
        with pytest.raises(ValueError, match="No 2D histogram data available"):
            api.apply_colormap(ndxplorer)

    def test_update_plots_both(self):
        """Test updating both histograms and colormap through API."""
        ndxplorer = Mock()
        
        with patch('ndxplorer.plotting.histograms.update_histogram_display') as mock_hist:
            with patch('ndxplorer.plotting.colormaps.update_colormap') as mock_cmap:
                api.update_plots(ndxplorer, update_histograms=True, update_colormap=True)
                
                mock_hist.assert_called_once_with(ndxplorer)
                mock_cmap.assert_called_once_with(ndxplorer)

    def test_update_plots_histograms_only(self):
        """Test updating only histograms through API."""
        ndxplorer = Mock()
        
        with patch('ndxplorer.plotting.histograms.update_histogram_display') as mock_hist:
            api.update_plots(ndxplorer, update_histograms=True, update_colormap=False)
            
            mock_hist.assert_called_once_with(ndxplorer)

    def test_update_plots_colormap_only(self):
        """Test updating only colormap through API."""
        ndxplorer = Mock()
        
        with patch('ndxplorer.plotting.colormaps.update_colormap') as mock_cmap:
            api.update_plots(ndxplorer, update_histograms=False, update_colormap=True)
            
            mock_cmap.assert_called_once_with(ndxplorer)

    def test_get_plot_statistics_histogram(self):
        """Test getting histogram statistics through API."""
        ndxplorer = Mock()
        
        with patch('ndxplorer.plotting.histograms.get_histogram_statistics') as mock_stats:
            mock_stats.return_value = {"count": 100, "mean": 50}
            
            result = api.get_plot_statistics(ndxplorer, plot_type="histogram", dimension="2d")
            
            mock_stats.assert_called_once_with(ndxplorer, "2d")
            assert result == {"count": 100, "mean": 50}

    def test_get_plot_statistics_scatter(self):
        """Test getting scatter statistics through API."""
        ndxplorer = Mock()
        ndxplorer.x_values = np.array([1, 2, 3, 4])
        ndxplorer.y_values = np.array([1, 2, 3, 4])
        
        with patch('ndxplorer.plotting.scatter.compute_scatter_statistics') as mock_stats:
            mock_stats.return_value = {"n_points": 4, "correlation": 1.0}
            
            result = api.get_plot_statistics(ndxplorer, plot_type="scatter")
            
            mock_stats.assert_called_once_with(ndxplorer, ndxplorer.x_values, ndxplorer.y_values)
            assert result == {"n_points": 4, "correlation": 1.0}

    def test_get_plot_statistics_invalid_type(self):
        """Test getting statistics with invalid plot type through API."""
        ndxplorer = Mock()
        
        with pytest.raises(ValueError, match="Unknown plot type"):
            api.get_plot_statistics(ndxplorer, plot_type="invalid")

    def test_export_plot_data_histogram(self):
        """Test exporting histogram data through API."""
        ndxplorer = Mock()
        
        with patch('ndxplorer.plotting.histograms.plot_histogram') as mock_hist:
            mock_hist.return_value = (np.array([1, 2, 3]), np.array([0, 1, 2, 3]))
            
            result = api.export_plot_data(ndxplorer, plot_type="histogram", dimension="x", format="csv")
            
            mock_hist.assert_called_once()
            assert "counts,bin_start,bin_end" in result

    def test_export_plot_data_scatter(self):
        """Test exporting scatter data through API."""
        ndxplorer = Mock()
        
        with patch('ndxplorer.plotting.scatter.create_scatter_plot') as mock_scatter:
            with patch('ndxplorer.plotting.scatter.export_scatter_data') as mock_export:
                mock_scatter.return_value = {"x": [1, 2, 3], "y": [4, 5, 6]}
                mock_export.return_value = "x,y\n1,4\n2,5\n3,6"
                
                result = api.export_plot_data(ndxplorer, plot_type="scatter", format="csv")
                
                mock_scatter.assert_called_once()
                mock_export.assert_called_once()
                assert result == "x,y\n1,4\n2,5\n3,6"

    def test_create_custom_plot_histogram(self):
        """Test creating custom histogram plot through API."""
        ndxplorer = Mock()
        data = np.array([1, 2, 3, 4, 5])
        
        with patch('ndxplorer.plotting.histograms.compute_1d_histogram') as mock_hist:
            mock_hist.return_value = (np.array([1, 1, 1, 1, 1]), np.array([0, 1, 2, 3, 4, 5]))
            
            result = api.create_custom_plot(ndxplorer, plot_type="custom_histogram", data=data)
            
            mock_hist.assert_called_once_with(ndxplorer, data)
            assert result is not None

    def test_create_custom_plot_scatter(self):
        """Test creating custom scatter plot through API."""
        ndxplorer = Mock()
        data = np.array([[1, 4], [2, 5], [3, 6]])
        
        with patch('ndxplorer.plotting.scatter.create_scatter_plot') as mock_scatter:
            mock_scatter.return_value = {"x": [1, 2, 3], "y": [4, 5, 6]}
            
            result = api.create_custom_plot(ndxplorer, plot_type="custom_scatter", data=data)
            
            mock_scatter.assert_called_once()
            assert result is not None

    def test_create_custom_plot_colormap(self):
        """Test creating custom colormap plot through API."""
        ndxplorer = Mock()
        data = np.array([[1, 2], [3, 4]])
        
        with patch('ndxplorer.plotting.colormaps.apply_colormap_to_data') as mock_cmap:
            mock_cmap.return_value = np.array([[[255, 0, 0, 255], [0, 255, 0, 255]]])
            
            result = api.create_custom_plot(ndxplorer, plot_type="custom_colormap", data=data)
            
            mock_cmap.assert_called_once()
            assert result is not None

    def test_create_custom_plot_invalid_type(self):
        """Test creating custom plot with invalid type through API."""
        ndxplorer = Mock()
        
        with pytest.raises(ValueError, match="Unknown custom plot type"):
            api.create_custom_plot(ndxplorer, plot_type="invalid")

    def test_convenience_functions(self):
        """Test convenience functions through API."""
        ndxplorer = Mock()
        
        with patch('ndxplorer.plotting.api.plot_histogram') as mock_hist:
            with patch('ndxplorer.plotting.api.plot_scatter') as mock_scatter:
                with patch('ndxplorer.plotting.api.apply_colormap') as mock_cmap:
                    with patch('ndxplorer.plotting.api.update_plots') as mock_update:
                        
                        # Test quick functions
                        api.quick_histogram(ndxplorer, "2d")
                        mock_hist.assert_called_with(ndxplorer, "2d")
                        
                        api.quick_scatter(ndxplorer)
                        mock_scatter.assert_called_with(ndxplorer)
                        
                        api.quick_colormap(ndxplorer)
                        mock_cmap.assert_called_with(ndxplorer)
                        
                        api.refresh_all_plots(ndxplorer)
                        mock_update.assert_called_with(ndxplorer, update_histograms=True, update_colormap=True)


if __name__ == "__main__":
    pytest.main([__file__])
