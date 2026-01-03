"""
Mask drawing and loading utilities for NDXplorer.

Provides functionality to:
- Draw masks with specified category/class values
- Load integer TIFF files as masks (different ints = different classes)
- Save masks as binary bitmaps
- Convert between mask formats
"""
from typing import Optional, Dict, List, Tuple
import numpy as np
from pathlib import Path

try:
    from chisurf.gui import QtCore, QtWidgets
except ImportError:
    from qtpy import QtCore, QtWidgets


def load_mask_from_tiff(filename: str) -> Tuple[np.ndarray, List[int]]:
    """
    Load a mask from an integer TIFF file.
    
    Different integer values in the TIFF correspond to different classes/categories.
    
    Parameters
    ----------
    filename : str
        Path to the TIFF file
        
    Returns
    -------
    mask : np.ndarray
        The mask array with integer class labels
    classes : List[int]
        List of unique class values found in the mask
    """
    try:
        import skimage.io as skio
    except ImportError:
        import tifffile
        mask = tifffile.imread(filename)
    else:
        mask = skio.imread(filename)
    
    # Ensure integer type
    if mask.dtype.kind == 'f':
        mask = mask.astype(np.int32)
    
    # Get unique classes (excluding 0 which is typically background)
    classes = sorted([int(c) for c in np.unique(mask) if c != 0])
    
    return mask, classes


def save_mask_as_bitmap(mask: np.ndarray, filename: str, binary: bool = True) -> None:
    """
    Save a mask as a binary bitmap or integer TIFF.
    
    Parameters
    ----------
    mask : np.ndarray
        The mask array
    filename : str
        Output filename
    binary : bool
        If True, save as binary (0/255). If False, save integer values as-is.
    """
    try:
        import skimage.io as skio
    except ImportError:
        import tifffile
        save_func = tifffile.imwrite
    else:
        save_func = skio.imsave
    
    output = mask.copy()
    
    if binary:
        # Convert to binary: any non-zero value becomes 255
        output = np.where(output > 0, 255, 0).astype(np.uint8)
    else:
        # Keep integer values but ensure proper dtype
        if output.max() <= 255:
            output = output.astype(np.uint8)
        else:
            output = output.astype(np.uint16)
    
    save_func(filename, output)


def create_empty_mask(shape: Tuple[int, int], dtype=np.int32) -> np.ndarray:
    """
    Create an empty mask with the specified shape.
    
    Parameters
    ----------
    shape : Tuple[int, int]
        Shape of the mask (height, width)
    dtype : numpy dtype
        Data type for the mask
        
    Returns
    -------
    mask : np.ndarray
        Empty mask filled with zeros
    """
    return np.zeros(shape, dtype=dtype)


def merge_masks(masks: Dict[int, np.ndarray]) -> np.ndarray:
    """
    Merge multiple binary masks into a single multi-class mask.
    
    Parameters
    ----------
    masks : Dict[int, np.ndarray]
        Dictionary mapping class IDs to binary masks
        
    Returns
    -------
    merged_mask : np.ndarray
        Single mask with integer class labels
    """
    if not masks:
        return None
    
    # Get shape from first mask
    shape = next(iter(masks.values())).shape
    merged = np.zeros(shape, dtype=np.int32)
    
    # Merge masks, with higher class IDs overwriting lower ones
    for class_id in sorted(masks.keys()):
        mask = masks[class_id]
        merged[mask > 0] = class_id
    
    return merged


def extract_class_mask(mask: np.ndarray, class_id: int) -> np.ndarray:
    """
    Extract a binary mask for a specific class from a multi-class mask.
    
    Parameters
    ----------
    mask : np.ndarray
        Multi-class mask with integer labels
    class_id : int
        Class ID to extract
        
    Returns
    -------
    binary_mask : np.ndarray
        Binary mask for the specified class
    """
    return (mask == class_id).astype(np.uint8)


def apply_brush_to_mask(
    mask: np.ndarray,
    pos: Tuple[int, int],
    brush_kernel: np.ndarray,
    class_id: int,
    mode: str = 'add'
) -> np.ndarray:
    """
    Apply a brush kernel to a mask at the specified position.
    
    Parameters
    ----------
    mask : np.ndarray
        The mask to modify
    pos : Tuple[int, int]
        Position (y, x) to apply the brush
    brush_kernel : np.ndarray
        Brush kernel to apply
    class_id : int
        Class ID to paint
    mode : str
        'add' to paint, 'erase' to remove
        
    Returns
    -------
    mask : np.ndarray
        Modified mask
    """
    y, x = pos
    kernel_h, kernel_w = brush_kernel.shape
    half_h, half_w = kernel_h // 2, kernel_w // 2
    
    # Calculate bounds
    y_start = max(0, y - half_h)
    y_end = min(mask.shape[0], y + half_h + 1)
    x_start = max(0, x - half_w)
    x_end = min(mask.shape[1], x + half_w + 1)
    
    # Calculate kernel bounds
    ky_start = max(0, half_h - y)
    ky_end = kernel_h - max(0, (y + half_h + 1) - mask.shape[0])
    kx_start = max(0, half_w - x)
    kx_end = kernel_w - max(0, (x + half_w + 1) - mask.shape[1])
    
    # Apply brush
    kernel_region = brush_kernel[ky_start:ky_end, kx_start:kx_end]
    
    if mode == 'add':
        # Paint with class_id where kernel is non-zero
        mask[y_start:y_end, x_start:x_end][kernel_region > 0] = class_id
    elif mode == 'erase':
        # Erase (set to 0) where kernel is non-zero
        mask[y_start:y_end, x_start:x_end][kernel_region > 0] = 0
    
    return mask


def mask_to_selection(mask: np.ndarray, class_id: Optional[int] = None) -> np.ndarray:
    """
    Convert a mask to a selection array suitable for NDXplorer.
    
    Parameters
    ----------
    mask : np.ndarray
        The mask array
    class_id : Optional[int]
        If specified, only select pixels with this class ID.
        If None, select all non-zero pixels.
        
    Returns
    -------
    selection : np.ndarray
        Boolean array indicating selected pixels
    """
    if class_id is not None:
        return mask == class_id
    else:
        return mask > 0


def get_mask_statistics(mask: np.ndarray) -> Dict[int, int]:
    """
    Get statistics about a mask.
    
    Parameters
    ----------
    mask : np.ndarray
        The mask array
        
    Returns
    -------
    stats : Dict[int, int]
        Dictionary mapping class IDs to pixel counts
    """
    unique, counts = np.unique(mask, return_counts=True)
    return {int(u): int(c) for u, c in zip(unique, counts) if u != 0}


def create_gaussian_brush_kernel(size: int = 21, sigma: float = 3.0) -> np.ndarray:
    """
    Create a Gaussian brush kernel.
    
    Parameters
    ----------
    size : int
        Size of the kernel (should be odd)
    sigma : float
        Standard deviation of the Gaussian
        
    Returns
    -------
    kernel : np.ndarray
        Gaussian kernel
    """
    import scipy.stats as st
    
    # Ensure odd size
    if size % 2 == 0:
        size += 1
    
    interval = (2.0 * sigma + 1.) / size
    x = np.linspace(-sigma - interval/2., sigma + interval/2., size + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    
    # Threshold to create binary brush
    kernel = (kernel > 0.001).astype(np.float32)
    
    return kernel


def create_circular_brush_kernel(radius: int = 5) -> np.ndarray:
    """
    Create a circular brush kernel.
    
    Parameters
    ----------
    radius : int
        Radius of the circle
        
    Returns
    -------
    kernel : np.ndarray
        Circular kernel
    """
    size = 2 * radius + 1
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    kernel = (x**2 + y**2 <= radius**2).astype(np.float32)
    return kernel


def create_pixel_radius_brush_kernel(
    radius_px: int,
    px_per_bin_x: float,
    px_per_bin_y: float,
) -> np.ndarray:
    radius_px = int(max(1, radius_px))
    px_per_bin_x = float(max(1e-6, px_per_bin_x))
    px_per_bin_y = float(max(1e-6, px_per_bin_y))

    rx = int(np.ceil(radius_px / px_per_bin_x))
    ry = int(np.ceil(radius_px / px_per_bin_y))

    y, x = np.ogrid[-ry:ry + 1, -rx:rx + 1]
    kernel = ((x * px_per_bin_x) ** 2 + (y * px_per_bin_y) ** 2 <= radius_px**2).astype(np.float32)
    return kernel
