"""LUT and color management functionality for NDXplorer.

This module provides colormap management, LUT operations, and color
mapping utilities extracted from plot_main.py.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union
import numpy as np

from ..logging_config import logging

if False:  # pragma: no cover - type checking hints without runtime import
    from ..core.plot_main import NDXplorer

try:
    import pyqtgraph as pg
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False
    pg = None
    logging.warning("pyqtgraph not available, colormap functionality limited")


def get_available_colormaps() -> List[str]:
    if PYQTGRAPH_AVAILABLE:
        try:
            return sorted(set(pg.colormap.listMaps()))
        except Exception as e:
            logging.warning(f"Failed to get colormap list from pyqtgraph: {e}")

    return ["viridis", "plasma", "inferno", "magma", "cividis"]


def create_colormap_lut(
    colormap_name: str,
    n_colors: int = 256,
    gamma: float = 1.0
) -> np.ndarray:
    """
    Create lookup table (LUT) for specified colormap.
    
    Args:
        colormap_name: Name of the colormap
        n_colors: Number of colors in LUT
        gamma: Gamma correction factor
        
    Returns:
        Array of RGBA color values (shape: n_colors, 4)
    """
    if PYQTGRAPH_AVAILABLE:
        try:
            cmap = pg.colormap.get(colormap_name)
            positions = np.linspace(0.0, 1.0, n_colors)
            colors = cmap.map(positions, mode="float")
            if gamma != 1.0:
                colors[:, :3] = np.power(colors[:, :3], gamma)
            return (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)
        except Exception as e:
            logging.warning(f"Failed to create pyqtgraph LUT for {colormap_name}: {e}")

    logging.warning("pyqtgraph colormap lookup failed for %s, using grayscale LUT", colormap_name)
    lut = np.zeros((n_colors, 4), dtype=np.uint8)
    lut[:, 0] = np.linspace(0, 255, n_colors)
    lut[:, 1] = np.linspace(0, 255, n_colors)
    lut[:, 2] = np.linspace(0, 255, n_colors)
    lut[:, 3] = 255
    return lut


def apply_colormap_to_data(
    data: np.ndarray,
    colormap_name: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    gamma: float = 1.0
) -> np.ndarray:
    """
    Apply colormap to 2D data array.
    
    Args:
        data: 2D data array
        colormap_name: Name of colormap to apply
        vmin: Minimum data value for mapping
        vmax: Maximum data value for mapping
        gamma: Gamma correction factor
        
    Returns:
        RGBA image array (shape: height, width, 4)
    """
    if data.ndim != 2:
        raise ValueError("Data must be 2-dimensional")
    
    # Determine data range
    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)
    
    # Handle case where vmin == vmax
    if vmin == vmax:
        vmin = vmin - 0.5
        vmax = vmax + 0.5
    
    # Normalize data to [0, 1]
    norm_data = (data - vmin) / (vmax - vmin)
    norm_data = np.clip(norm_data, 0, 1)
    
    # Apply gamma correction
    if gamma != 1.0:
        norm_data = np.power(norm_data, gamma)
    
    if PYQTGRAPH_AVAILABLE:
        try:
            cmap = pg.colormap.get(colormap_name)
            colored_data = cmap.map(norm_data, mode="float")
            return (np.clip(colored_data, 0.0, 1.0) * 255).astype(np.uint8)
        except Exception as e:
            logging.warning(f"Failed to apply pyqtgraph colormap '{colormap_name}': {e}")

    logging.warning("pyqtgraph colormap %s failed, using grayscale", colormap_name)
    gray_data = (norm_data * 255).astype(np.uint8)
    rgba_data = np.stack([gray_data, gray_data, gray_data, np.full_like(gray_data, 255)], axis=-1)
    return rgba_data


def get_colormap_limits(
    data: np.ndarray,
    percentile_range: Tuple[float, float] = (1.0, 99.0)
) -> Tuple[float, float]:
    """
    Compute optimal colormap limits for data.
    
    Args:
        data: Input data array
        percentile_range: Percentile range for auto-scaling
        
    Returns:
        Tuple of (vmin, vmax)
    """
    if data.size == 0:
        return 0.0, 1.0
    
    # Remove NaN/Inf values
    finite_data = data[np.isfinite(data)]
    
    if finite_data.size == 0:
        return 0.0, 1.0
    
    if percentile_range == (0, 100):
        vmin, vmax = np.min(finite_data), np.max(finite_data)
    else:
        vmin = np.percentile(finite_data, percentile_range[0])
        vmax = np.percentile(finite_data, percentile_range[1])
    
    # Ensure vmin < vmax
    if vmin == vmax:
        vmin = vmin - 0.5
        vmax = vmax + 0.5
    
    return float(vmin), float(vmax)


def create_custom_colormap(
    colors: List[Union[str, Tuple[float, float, float]]],
    name: str = "custom",
    positions: Optional[List[float]] = None
) -> str:
    """
    Create custom colormap from list of colors.
    
    Args:
        colors: List of color names or RGB tuples
        name: Name for the custom colormap
        positions: Optional positions for colors (0-1)
        
    Returns:
        Name of the created colormap
    """
    try:
        if positions is None:
            positions = np.linspace(0, 1, len(colors))

        rgb_colors = []
        for color in colors:
            if isinstance(color, str):
                qcolor = pg.mkColor(color) if PYQTGRAPH_AVAILABLE else None
                if qcolor is None:
                    raise ValueError(f"Named color requires pyqtgraph: {color}")
                rgb_colors.append((qcolor.red(), qcolor.green(), qcolor.blue(), qcolor.alpha()))
            else:
                rgb = tuple(color)
                if max(rgb) <= 1.0:
                    rgb = tuple(int(component * 255) for component in rgb)
                rgb_colors.append(rgb)

        pg.colormap.ColorMap(np.asarray(positions, dtype=float), np.asarray(rgb_colors, dtype=np.ubyte))
        return name

    except Exception:
        logging.warning("pyqtgraph not available, custom colormap creation failed")
        return "viridis"


def current_cmap(ndxplorer: "NDXplorer") -> str:
    """Get current colormap name from combobox."""
    if hasattr(ndxplorer, 'comboBoxCmap'):
        return ndxplorer.comboBoxCmap.currentText()
    return "viridis"  # fallback


def populate_colormap_combobox(ndxplorer: "NDXplorer") -> None:
    """
    Populate colormap combobox with available colormaps.
    
    Args:
        ndxplorer: NDXplorer instance with comboBoxCmap attribute
    """
    if not hasattr(ndxplorer, 'comboBoxCmap'):
        logging.warning("NDXplorer instance missing comboBoxCmap")
        return
    
    colormap_names = get_available_colormaps()
    logging.debug(f"Found {len(colormap_names)} colormaps")
    
    # Clear existing items
    ndxplorer.comboBoxCmap.clear()
    
    # Add colormaps
    ndxplorer.comboBoxCmap.addItems(colormap_names)
    
    # Set default if available
    default_cmap = "viridis"
    if default_cmap in colormap_names:
        index = colormap_names.index(default_cmap)
        ndxplorer.comboBoxCmap.setCurrentIndex(index)
        logging.info(f"Set default colormap to {default_cmap}")


def update_colormap(ndxplorer: "NDXplorer", colormap_name: Optional[str] = None) -> bool:
    """Update the colormap of the 2D plot image item."""
    try:
        if not getattr(ndxplorer, "_deferred_init_done", False) or ndxplorer.g_2dplot is None:
            logging.debug("Plot not ready for colormap update")
            return False

        if colormap_name is None:
            colormap_name = current_cmap(ndxplorer)

        vmin = getattr(ndxplorer, "vmin", 0.0)
        vmax = getattr(ndxplorer, "vmax", 1.0)
        ndxplorer.cax.set_colormap(colormap_name, vmin, vmax)
        ndxplorer.g_2dplot.replot()
        logging.debug(f"Updated pyqtgraph colormap to {colormap_name}")
        return True

    except Exception as e:
        logging.error(f"Failed to update colormap: {e}")
        return False


update_guiqwt_colormap = update_colormap


def get_colormap_statistics(
    colored_data: np.ndarray
) -> dict:
    """
    Compute statistics for colored/image data.
    
    Args:
        colored_data: RGBA image array
        
    Returns:
        Dictionary containing color statistics
    """
    if colored_data.ndim != 3 or colored_data.shape[2] != 4:
        raise ValueError("Data must be RGBA image array")
    
    # Separate channels
    r, g, b, a = colored_data[:, :, 0], colored_data[:, :, 1], colored_data[:, :, 2], colored_data[:, :, 3]
    
    stats = {
        "shape": colored_data.shape,
        "red_mean": np.mean(r),
        "green_mean": np.mean(g),
        "blue_mean": np.mean(b),
        "alpha_mean": np.mean(a),
        "red_std": np.std(r),
        "green_std": np.std(g),
        "blue_std": np.std(b),
        "alpha_std": np.std(a)
    }
    
    return stats


def set_default_colormap(ndxplorer: "NDXplorer", default_cmap: str) -> None:
    """
    Set default colormap in combobox and update plot if ready.
    
    Args:
        ndxplorer: NDXplorer instance
        default_cmap: Name of default colormap
    """
    logging.info(f"Setting default colormap to {default_cmap}")
    
    if not hasattr(ndxplorer, 'comboBoxCmap'):
        logging.warning("NDXplorer instance missing comboBoxCmap")
        return
    
    index = ndxplorer.comboBoxCmap.findText(default_cmap)
    if index == -1:
        logging.info(f"Colormap {default_cmap} not found in available colormaps")
        return
    
    ndxplorer.comboBoxCmap.setCurrentIndex(index)
    
    if getattr(ndxplorer, "_deferred_init_done", False) and hasattr(ndxplorer, 'cax') and ndxplorer.cax is not None:
        vmin = getattr(ndxplorer, 'vmin', 0.0)
        vmax = getattr(ndxplorer, 'vmax', 1.0)
        ndxplorer.cax.set_colormap(default_cmap, vmin, vmax)
        if hasattr(ndxplorer, 'g_2dplot') and ndxplorer.g_2dplot is not None:
            ndxplorer.g_2dplot.replot()
