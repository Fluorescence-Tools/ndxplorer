"""Unified plotting API for NDXplorer.

This module provides a high-level interface for all plotting operations,
wrapping the specialized modules (histograms, scatter, colormaps) into
a consistent API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from . import histograms, scatter, colormaps

if False:  # pragma: no cover - type checking hints without runtime import
    from ..core.plot_main import NDXplorer


def plot_histogram(
    ndxplorer: "NDXplorer",
    dimension: str = "2d",
    **kwargs
) -> Tuple[np.ndarray, Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
    """
    Plot histogram data.
    
    Args:
        ndxplorer: NDXplorer instance
        dimension: Histogram dimension ("x", "y", "z", or "2d")
        **kwargs: Additional plotting options
        
    Returns:
        Tuple of (histogram_data, bin_edges)
    """
    return histograms.plot_histogram(ndxplorer, dimension, **kwargs)


def plot_scatter(
    ndxplorer: "NDXplorer",
    plot_type: str = "standard",
    x_data: Optional[np.ndarray] = None,
    y_data: Optional[np.ndarray] = None,
    color_data: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create scatter plot.
    
    Args:
        ndxplorer: NDXplorer instance
        plot_type: Type of scatter plot ("standard", "weighted", "density")
        x_data: Optional x-axis data (uses ndxplorer.x_values if None)
        y_data: Optional y-axis data (uses ndxplorer.y_values if None)
        color_data: Optional color data
        weights: Optional weights for weighted scatter
        **kwargs: Additional plotting options
        
    Returns:
        Dictionary containing plot data
    """
    if x_data is None:
        x_data = ndxplorer.x_values
    if y_data is None:
        y_data = ndxplorer.y_values
    
    if plot_type == "weighted":
        return scatter.create_weighted_scatter(ndxplorer, x_data, y_data, weights, **kwargs)
    elif plot_type == "density":
        return scatter.create_density_scatter(ndxplorer, x_data, y_data, **kwargs)
    else:
        size_data = kwargs.pop("size_data", None)
        return scatter.create_scatter_plot(
            ndxplorer,
            x_data,
            y_data,
            color_data,
            size_data,
            weights,
            **kwargs,
        )


def apply_colormap(
    ndxplorer: "NDXplorer",
    data: Optional[np.ndarray] = None,
    colormap_name: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    **kwargs
) -> np.ndarray:
    """
    Apply colormap to data.
    
    Args:
        ndxplorer: NDXplorer instance
        data: Data to colorize (uses 2D histogram if None)
        colormap_name: Name of colormap (uses current if None)
        vmin: Minimum value for mapping
        vmax: Maximum value for mapping
        **kwargs: Additional colormap options
        
    Returns:
        RGBA image array
    """
    if data is None:
        if "2d" not in ndxplorer._histogram:
            raise ValueError("No 2D histogram data available")
        hist_2d = ndxplorer._histogram["2d"]
        if hasattr(hist_2d, 'H'):
            data = hist_2d.H
        else:
            data = hist_2d[0]
    
    if colormap_name is None:
        colormap_name = colormaps.current_cmap(ndxplorer)
    
    return colormaps.apply_colormap_to_data(data, colormap_name, vmin, vmax, **kwargs)


def update_plots(
    ndxplorer: "NDXplorer",
    update_histograms: bool = True,
    update_colormap: bool = True,
    **kwargs
) -> None:
    """
    Update all plot displays.
    
    Args:
        ndxplorer: NDXplorer instance
        update_histograms: Whether to update histogram displays
        update_colormap: Whether to update colormap
        **kwargs: Additional update options
    """
    if update_histograms:
        histograms.update_histogram_display(ndxplorer, **kwargs)
    
    if update_colormap:
        colormaps.update_colormap(ndxplorer, **kwargs)


def get_plot_statistics(
    ndxplorer: "NDXplorer",
    plot_type: str,
    dimension: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Get statistics for plot data.
    
    Args:
        ndxplorer: NDXplorer instance
        plot_type: Type of plot ("histogram" or "scatter")
        dimension: Histogram dimension (required for histogram plots)
        **kwargs: Additional statistics options
        
    Returns:
        Dictionary containing statistics
    """
    if plot_type == "histogram":
        if dimension is None:
            raise ValueError("Dimension required for histogram statistics")
        return histograms.get_histogram_statistics(ndxplorer, dimension)
    elif plot_type == "scatter":
        x_data = kwargs.get("x_data", ndxplorer.x_values)
        y_data = kwargs.get("y_data", ndxplorer.y_values)
        return scatter.compute_scatter_statistics(ndxplorer, x_data, y_data)
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")


def export_plot_data(
    ndxplorer: "NDXplorer",
    plot_type: str,
    format: str = "csv",
    filename: Optional[str] = None,
    dimension: Optional[str] = None,
    **kwargs
) -> Optional[str]:
    """
    Export plot data to file or string.
    
    Args:
        ndxplorer: NDXplorer instance
        plot_type: Type of plot to export
        format: Export format ('csv', 'json', 'numpy')
        filename: Optional output filename
        dimension: Histogram dimension (required for histogram plots)
        **kwargs: Additional export options
        
    Returns:
        String representation if filename is None, otherwise None
    """
    if plot_type == "histogram":
        if dimension is None:
            raise ValueError("Dimension required for histogram export")
        
        hist_data, bin_edges = histograms.plot_histogram(ndxplorer, dimension, **kwargs)
        
        if dimension == "2d":
            # 2D histogram export
            if format == "numpy":
                if filename:
                    np.savez(filename, histogram=hist_data)
                    return None
                else:
                    return {"histogram": hist_data}
            else:
                # CSV/JSON export for 2D
                if format == "csv":
                    csv_data = "x,y,value\n"
                    hist_2d = ndxplorer._histogram["2d"]
                    if hasattr(hist_2d, 'x_edges'):
                        x_edges, y_edges = hist_2d.x_edges, hist_2d.y_edges
                    else:
                        x_edges, y_edges = hist_2d[1], hist_2d[2]
                    for i in range(hist_data.shape[0]):
                        for j in range(hist_data.shape[1]):
                            csv_data += f"{x_edges[i]},{y_edges[j]},{hist_data[i,j]}\n"
                    if filename:
                        with open(filename, 'w') as f:
                            f.write(csv_data)
                        return None
                    return csv_data
        else:
            # 1D histogram export
            if format == "numpy":
                if filename:
                    np.savez(filename, histogram=hist_data, bins=bin_edges)
                    return None
                else:
                    return {"histogram": hist_data, "bins": bin_edges}
            else:
                if format == "csv":
                    csv_data = "counts,bin_start,bin_end\n"
                    for count, bin_start, bin_end in zip(hist_data, bin_edges[:-1], bin_edges[1:]):
                        csv_data += f"{count},{bin_start},{bin_end}\n"
                    if filename:
                        with open(filename, 'w') as f:
                            f.write(csv_data)
                        return None
                    return csv_data
    
    elif plot_type == "scatter":
        scatter_data = scatter.create_scatter_plot(ndxplorer, **kwargs)
        return scatter.export_scatter_data(scatter_data, format, filename)
    
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")


def create_custom_plot(
    ndxplorer: "NDXplorer",
    plot_type: str,
    data: Optional[np.ndarray] = None,
    **kwargs
) -> Any:
    """
    Create custom plot with user-provided data.
    
    Args:
        ndxplorer: NDXplorer instance
        plot_type: Type of custom plot
        data: Input data
        **kwargs: Additional plotting options
        
    Returns:
        Plot result
    """
    known_plot_types = {"custom_histogram", "custom_scatter", "custom_colormap"}
    if plot_type not in known_plot_types:
        raise ValueError(f"Unknown custom plot type: {plot_type}")
    if data is None:
        raise ValueError("Data required for custom plot")

    if plot_type == "custom_histogram":
        return histograms.compute_1d_histogram(ndxplorer, data, **kwargs)
    elif plot_type == "custom_scatter":
        if data.ndim == 2 and data.shape[1] >= 2:
            return scatter.create_scatter_plot(ndxplorer, data[:, 0], data[:, 1], **kwargs)
        else:
            raise ValueError("Scatter data must be 2D with at least 2 columns")
    elif plot_type == "custom_colormap":
        return colormaps.apply_colormap_to_data(data, **kwargs)


# Convenience functions for common operations
def quick_histogram(ndxplorer: "NDXplorer", dimension: str = "2d") -> Any:
    """Quick histogram plot with default settings."""
    return plot_histogram(ndxplorer, dimension)


def quick_scatter(ndxplorer: "NDXplorer") -> Dict[str, Any]:
    """Quick scatter plot with default settings."""
    return plot_scatter(ndxplorer)


def quick_colormap(ndxplorer: "NDXplorer") -> np.ndarray:
    """Quick colormap application with default settings."""
    return apply_colormap(ndxplorer)


def refresh_all_plots(ndxplorer: "NDXplorer") -> None:
    """Refresh all plot displays."""
    update_plots(ndxplorer, update_histograms=True, update_colormap=True)


# Export all public functions
__all__ = [
    "plot_histogram", "plot_scatter", "apply_colormap", "update_plots",
    "get_plot_statistics", "export_plot_data", "create_custom_plot",
    "quick_histogram", "quick_scatter", "quick_colormap", "refresh_all_plots"
]
