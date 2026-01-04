"""Scatter plot functionality for NDXplorer.

This module provides scatter plot capabilities including weighted plots,
color mapping, and interactive selection features.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union
import numpy as np

from ..logging_config import logging

if False:  # pragma: no cover - type checking hints without runtime import
    from ..core.plot_main import NDXplorer


def create_scatter_plot(
    ndxplorer: "NDXplorer",
    x_data: np.ndarray,
    y_data: np.ndarray,
    color_data: Optional[np.ndarray] = None,
    size_data: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    alpha: float = 0.7,
    **kwargs
) -> dict:
    """
    Create scatter plot data with optional color and size mapping.
    
    Args:
        ndxplorer: NDXplorer instance
        x_data: X-axis data
        y_data: Y-axis data
        color_data: Optional data for color mapping
        size_data: Optional data for size mapping
        weights: Optional weight array
        alpha: Transparency value (0-1)
        **kwargs: Additional plotting options
        
    Returns:
        Dictionary containing plot data and metadata
    """
    # Validate input data
    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data must have the same length")
    
    n_points = len(x_data)
    
    # Handle color mapping
    if color_data is None:
        colors = np.full(n_points, 0.5)  # Default color value
    else:
        if len(color_data) != n_points:
            raise ValueError("color_data must have the same length as x_data")
        colors = color_data
    
    # Handle size mapping
    if size_data is None:
        sizes = np.full(n_points, 20)  # Default size
    else:
        if len(size_data) != n_points:
            raise ValueError("size_data must have the same length as x_data")
        sizes = size_data
    
    # Apply weights if provided
    if weights is not None:
        if len(weights) != n_points:
            raise ValueError("weights must have the same length as x_data")
        # Normalize weights to influence alpha or size
        normalized_weights = weights / np.max(weights) if np.max(weights) > 0 else weights
        alpha = alpha * (0.3 + 0.7 * normalized_weights)  # Weighted transparency
    
    plot_data = {
        "x": x_data,
        "y": y_data,
        "colors": colors,
        "sizes": sizes,
        "alpha": alpha,
        "n_points": n_points,
        "has_color_mapping": color_data is not None,
        "has_size_mapping": size_data is not None,
        "has_weights": weights is not None
    }
    
    return plot_data


def create_weighted_scatter(
    ndxplorer: "NDXplorer",
    x_data: np.ndarray,
    y_data: np.ndarray,
    weights: np.ndarray,
    colormap: str = "viridis",
    size_scale: float = 50.0
) -> dict:
    """
    Create weighted scatter plot with color and size based on weights.
    
    Args:
        ndxplorer: NDXplorer instance
        x_data: X-axis data
        y_data: Y-axis data
        weights: Weight array for color and size mapping
        colormap: Name of colormap to use
        size_scale: Scale factor for point sizes
        
    Returns:
        Dictionary containing weighted plot data
    """
    if len(x_data) != len(y_data) or len(x_data) != len(weights):
        raise ValueError("All input arrays must have the same length")
    
    # Normalize weights for color mapping
    norm_weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights)) if np.max(weights) > np.min(weights) else np.zeros_like(weights)
    
    # Scale sizes based on weights
    sizes = 10 + norm_weights * size_scale
    
    return create_scatter_plot(
        ndxplorer=ndxplorer,
        x_data=x_data,
        y_data=y_data,
        color_data=norm_weights,
        size_data=sizes,
        weights=weights,
        colormap=colormap
    )


def apply_selection_to_scatter(
    plot_data: dict,
    selection_mask: np.ndarray,
    unselected_alpha: float = 0.2
) -> dict:
    """
    Apply selection mask to scatter plot data.
    
    Args:
        plot_data: Original scatter plot data
        selection_mask: Boolean mask for selected points
        unselected_alpha: Alpha value for unselected points
        
    Returns:
        Modified plot data with selection applied
    """
    if len(selection_mask) != plot_data["n_points"]:
        raise ValueError("selection_mask length must match number of points")
    
    modified_data = plot_data.copy()
    
    # Create alpha array based on selection
    alpha_array = np.full(plot_data["n_points"], unselected_alpha)
    alpha_array[selection_mask] = plot_data["alpha"]
    
    modified_data["alpha_array"] = alpha_array
    modified_data["selection"] = selection_mask
    modified_data["n_selected"] = np.sum(selection_mask)
    
    return modified_data


def compute_scatter_statistics(
    ndxplorer: "NDXplorer",
    x_data: np.ndarray,
    y_data: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> dict:
    """
    Compute statistics for scatter plot data.
    
    Args:
        ndxplorer: NDXplorer instance
        x_data: X-axis data
        y_data: Y-axis data
        weights: Optional weight array
        
    Returns:
        Dictionary containing scatter plot statistics
    """
    stats = {
        "n_points": len(x_data),
        "x_range": (np.min(x_data), np.max(x_data)),
        "y_range": (np.min(y_data), np.max(y_data)),
        "x_mean": np.mean(x_data),
        "y_mean": np.mean(y_data),
        "x_std": np.std(x_data),
        "y_std": np.std(y_data)
    }
    
    if weights is not None:
        weighted_x_mean = np.average(x_data, weights=weights)
        weighted_y_mean = np.average(y_data, weights=weights)
        stats.update({
            "weighted_x_mean": weighted_x_mean,
            "weighted_y_mean": weighted_y_mean,
            "total_weight": np.sum(weights)
        })
    
    # Compute correlation
    if len(x_data) > 1:
        correlation = np.corrcoef(x_data, y_data)[0, 1]
        stats["correlation"] = correlation if np.isfinite(correlation) else 0.0
    
    return stats


def create_density_scatter(
    ndxplorer: "NDXplorer",
    x_data: np.ndarray,
    y_data: np.ndarray,
    grid_size: int = 50,
    bandwidth: Optional[float] = None
) -> dict:
    """
    Create density-based scatter plot for large datasets.
    
    Args:
        ndxplorer: NDXplorer instance
        x_data: X-axis data
        y_data: Y-axis data
        grid_size: Size of density grid
        bandwidth: KDE bandwidth (auto-computed if None)
        
    Returns:
        Dictionary containing density scatter data
    """
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        logging.warning("scipy not available, falling back to regular scatter")
        return create_scatter_plot(ndxplorer, x_data, y_data)
    
    # Compute point density
    xy = np.vstack([x_data, y_data])
    
    try:
        kde = gaussian_kde(xy, bw_method=bandwidth)
        density = kde(xy)
        
        return create_scatter_plot(
            ndxplorer=ndxplorer,
            x_data=x_data,
            y_data=y_data,
            color_data=density,
            size_data=10 + density * 40  # Scale sizes by density
        )
    except Exception as e:
        logging.warning(f"Failed to compute density: {e}, using regular scatter")
        return create_scatter_plot(ndxplorer, x_data, y_data)


def export_scatter_data(
    plot_data: dict,
    format: str = "csv",
    filename: Optional[str] = None
) -> Union[str, None]:
    """
    Export scatter plot data to file or string.
    
    Args:
        plot_data: Scatter plot data dictionary
        format: Export format ('csv', 'json', 'numpy')
        filename: Optional output filename
        
    Returns:
        String data if filename is None, otherwise None
    """
    import pandas as pd
    
    # Create DataFrame
    df_data = {
        "x": plot_data["x"],
        "y": plot_data["y"],
        "colors": plot_data["colors"],
        "sizes": plot_data["sizes"]
    }
    
    if "alpha_array" in plot_data:
        df_data["alpha"] = plot_data["alpha_array"]
    
    df = pd.DataFrame(df_data)
    
    if format == "csv":
        output = df.to_csv(index=False)
    elif format == "json":
        output = df.to_json(orient="records")
    elif format == "numpy":
        output = {key: np.array(val) for key, val in df_data.items()}
        if filename:
            np.savez(filename, **output)
            return None
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    if filename:
        with open(filename, 'w') as f:
            f.write(output)
        return None
    else:
        return output
