"""
Mask drawing widget for NDXplorer.

Provides UI controls for drawing, loading, and saving masks with category/class specification.
"""
from typing import Optional, Dict
import numpy as np
from pathlib import Path

try:
    from chisurf.gui import QtCore, QtWidgets, QtGui
except ImportError:
    from qtpy import QtCore, QtWidgets, QtGui

from ..utils import mask_helpers


class MaskDrawingWidget(QtWidgets.QWidget):
    """
    Widget for mask drawing controls.
    
    Provides:
    - Category/class selection
    - Brush size control
    - Load/save mask functionality
    - Clear mask
    - Drawing mode (add/erase)
    """
    
    mask_changed = QtCore.Signal(np.ndarray)
    category_changed = QtCore.Signal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._mask = None
        self._current_category = 1
        self._brush_size = 5
        self._brush_kernel = None
        self._brush_px_per_bin_x = 1.0
        self._brush_px_per_bin_y = 1.0
        self._drawing_mode = 'add'
        self._mask_shape = None
        
        # UI elements will be set by plot_control.py after loading the .ui file
        self.category_spinbox = None
        self.brush_spinbox = None
        self.draw_radio = None
        self.erase_radio = None
        self.load_mask_btn = None
        self.save_mask_btn = None
        self.save_binary_btn = None
        self.clear_mask_btn = None
        self.apply_mask_btn = None
        self.enable_drawing_checkbox = None
        self.stats_label = None
        
        self._update_brush_kernel()
    
    def _on_category_changed(self, value: int):
        """Handle category change."""
        self._current_category = value
        self.category_changed.emit(value)
    
    def _on_brush_size_changed(self, value: int):
        """Handle brush size change."""
        self._brush_size = value
        self._update_brush_kernel()
    
    def _on_mode_changed(self):
        """Handle drawing mode change."""
        self._drawing_mode = 'add' if self.draw_radio.isChecked() else 'erase'
    
    def _update_brush_kernel(self):
        """Update the brush kernel based on current size."""
        self._brush_kernel = mask_helpers.create_pixel_radius_brush_kernel(
            self._brush_size,
            self._brush_px_per_bin_x,
            self._brush_px_per_bin_y,
        )

    def set_brush_pixel_scale(self, px_per_bin_x: float, px_per_bin_y: float):
        self._brush_px_per_bin_x = float(px_per_bin_x)
        self._brush_px_per_bin_y = float(px_per_bin_y)
        self._update_brush_kernel()
    
    def set_mask_shape(self, shape: tuple):
        """Set the shape for new masks."""
        self._mask_shape = shape
        if self._mask is None or self._mask.shape != shape:
            self._mask = mask_helpers.create_empty_mask(shape)
    
    def get_mask(self) -> Optional[np.ndarray]:
        """Get the current mask."""
        return self._mask
    
    def set_mask(self, mask: np.ndarray):
        """Set the current mask."""
        self._mask = mask.copy() if mask is not None else None
        self._update_statistics()
        self.mask_changed.emit(self._mask)
    
    def clear_mask(self):
        """Clear the current mask."""
        if self._mask_shape is not None:
            self._mask = mask_helpers.create_empty_mask(self._mask_shape)
            self._update_statistics()
            self.mask_changed.emit(self._mask)
    
    def apply_brush(self, pos: tuple, silent: bool = False):
        """
        Apply brush at the specified position.
        
        Parameters
        ----------
        pos : tuple
            Position (y, x) to apply brush
        silent : bool
            If True, don't emit mask_changed signal (for batch operations)
        """
        if self._mask is None:
            return
        
        self._mask = mask_helpers.apply_brush_to_mask(
            self._mask,
            pos,
            self._brush_kernel,
            self._current_category,
            self._drawing_mode
        )
        self._update_statistics()
        if not silent:
            self.mask_changed.emit(self._mask)
    
    def emit_mask_changed(self):
        """Emit mask_changed signal (for deferred updates after drawing)."""
        if self._mask is not None:
            self.mask_changed.emit(self._mask)
    
    def load_mask(self):
        """Load a mask from an integer TIFF file."""
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Mask",
            "",
            "TIFF files (*.tif *.tiff);;All files (*.*)"
        )
        
        if filename:
            try:
                mask, classes = mask_helpers.load_mask_from_tiff(filename)
                
                # Ensure mask matches current shape if set
                if self._mask_shape is not None and mask.shape != self._mask_shape:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Shape Mismatch",
                        f"Mask shape {mask.shape} does not match image shape {self._mask_shape}.\n"
                        "The mask will be loaded but may not align correctly."
                    )
                
                self._mask = mask
                self._mask_shape = mask.shape
                self._update_statistics()
                self.mask_changed.emit(self._mask)
                
                # Show info about loaded classes
                if classes:
                    QtWidgets.QMessageBox.information(
                        self,
                        "Mask Loaded",
                        f"Loaded mask with {len(classes)} classes: {classes}"
                    )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error Loading Mask",
                    f"Failed to load mask: {str(e)}"
                )
    
    def load_as_category(self):
        """Load an image and assign all non-zero pixels to current category."""
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Image as Category",
            "",
            "TIFF files (*.tif *.tiff);;All files (*.*)"
        )
        
        if filename:
            try:
                mask, _ = mask_helpers.load_mask_from_tiff(filename)
                
                # Convert to current category
                mask = np.where(mask > 0, self._current_category, 0).astype(np.int32)
                
                # Ensure mask matches current shape if set
                if self._mask_shape is not None and mask.shape != self._mask_shape:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Shape Mismatch",
                        f"Mask shape {mask.shape} does not match image shape {self._mask_shape}.\n"
                        "The mask will be loaded but may not align correctly."
                    )
                
                self._mask = mask
                self._mask_shape = mask.shape
                self._update_statistics()
                self.mask_changed.emit(self._mask)
                
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error Loading Image",
                    f"Failed to load image: {str(e)}"
                )
    
    def save_mask(self):
        """Save the current mask as an integer TIFF."""
        if self._mask is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No Mask",
                "No mask to save."
            )
            return
        
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Mask",
            "",
            "TIFF files (*.tif *.tiff);;All files (*.*)"
        )
        
        if filename:
            try:
                mask_helpers.save_mask_as_bitmap(self._mask, filename, binary=False)
                QtWidgets.QMessageBox.information(
                    self,
                    "Mask Saved",
                    f"Mask saved to {filename}"
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error Saving Mask",
                    f"Failed to save mask: {str(e)}"
                )
    
    def save_binary(self):
        """Save the current mask as a binary bitmap."""
        if self._mask is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No Mask",
                "No mask to save."
            )
            return
        
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Binary Mask",
            "",
            "TIFF files (*.tif *.tiff);;PNG files (*.png);;All files (*.*)"
        )
        
        if filename:
            try:
                mask_helpers.save_mask_as_bitmap(self._mask, filename, binary=True)
                QtWidgets.QMessageBox.information(
                    self,
                    "Mask Saved",
                    f"Binary mask saved to {filename}"
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error Saving Mask",
                    f"Failed to save mask: {str(e)}"
                )
    
    def _update_statistics(self):
        """Update the statistics label."""
        # Skip if stats_label is None (removed from UI)
        if self.stats_label is None:
            return
            
        if self._mask is None:
            self.stats_label.setText("")
            return
        
        stats = mask_helpers.get_mask_statistics(self._mask)
        if stats:
            total_pixels = sum(stats.values())
            stats_text = f"Pixels: {total_pixels} | Classes: {', '.join(f'{k}:{v}' for k, v in sorted(stats.items()))}"
            self.stats_label.setText(stats_text)
        else:
            self.stats_label.setText("Mask is empty")
    
    def get_current_category(self) -> int:
        """Get the current drawing category."""
        return self._current_category
    
    def get_brush_kernel(self) -> np.ndarray:
        """Get the current brush kernel."""
        return self._brush_kernel
    
    def get_drawing_mode(self) -> str:
        """Get the current drawing mode ('add' or 'erase')."""
        return self._drawing_mode
    
    def _on_apply_mask(self):
        """Handle apply mask button click."""
        # Use the stored direct reference to NDXplorer parent
        ndxplorer = getattr(self, '_ndxplorer_parent', None)
        
        if ndxplorer is not None and hasattr(ndxplorer, 'apply_mask_to_selection'):
            # Get current category
            category = self._current_category
            ndxplorer.apply_mask_to_selection(category)
        else:
            QtWidgets.QMessageBox.warning(
                self,
                "Error",
                "Could not find parent NDXplorer window."
            )
    
    def _on_drawing_enabled_changed(self, enabled: bool):
        """Handle drawing enabled groupbox state change."""
        try:
            from chisurf.gui import QtWidgets
            from ..logging_config import logging
            
            logging.info(f"Drawing enabled changed: {enabled}")
            
            # Use the stored direct reference to NDXplorer parent
            ndxplorer = getattr(self, '_ndxplorer_parent', None)
            
            if ndxplorer is None:
                logging.error("NDXplorer parent reference not set")
                QtWidgets.QMessageBox.warning(
                    self,
                    "Error",
                    "NDXplorer parent reference not found. Widget may not be properly initialized."
                )
                return
            
            if not hasattr(ndxplorer, 'mask_drawing'):
                logging.error("NDXplorer parent does not have mask_drawing attribute")
                QtWidgets.QMessageBox.warning(
                    self,
                    "Error",
                    "Mask drawing system not found in NDXplorer."
                )
                return
            
            if ndxplorer.mask_drawing is None:
                logging.error("NDXplorer mask_drawing is None - initialization may have failed")
                QtWidgets.QMessageBox.warning(
                    self,
                    "Error",
                    "Mask drawing system not initialized. Check logs for errors."
                )
                return
            
            logging.info(f"Toggling drawing mode to {enabled}")
            if enabled:
                ndxplorer.mask_drawing.enable_drawing_mode()
            else:
                ndxplorer.mask_drawing.disable_drawing_mode()
                
        except Exception as e:
            from ..logging_config import logging
            logging.error(f"Error in _on_drawing_enabled_changed: {e}", exc_info=True)
    
    def is_drawing_enabled(self) -> bool:
        """Check if drawing is enabled."""
        try:
            return self.enable_drawing_checkbox.isChecked()
        except (RuntimeError, AttributeError) as e:
            # Handle case where the widget might be deleted or not yet initialized
            from ..logging_config import logging
            logging.warning(f"Error checking drawing enabled state: {e}")
            return False
    
    def set_drawing_enabled(self, enabled: bool):
        """Set the drawing enabled state."""
        try:
            self.enable_drawing_checkbox.setChecked(enabled)
        except (RuntimeError, AttributeError) as e:
            # Handle case where the widget might be deleted or not yet initialized
            from ..logging_config import logging
            logging.warning(f"Error setting drawing enabled state: {e}")
