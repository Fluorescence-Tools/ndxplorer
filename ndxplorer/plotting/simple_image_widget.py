"""
Simple Qt-based image widget for displaying 2D histograms.
Replaces guiqwt-based plot with a lightweight QLabel-based implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple
import numpy as np
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtGui import QImage, QPixmap, QPainter, QFont, QPen, QColor
from qtpy.QtCore import Qt, QRect, QPoint

from ..logging_config import logging

if TYPE_CHECKING:
    from ..core.plot_main import NDXplorer


class SimpleImageWidget(QtWidgets.QWidget):
    """
    A simple widget for displaying 2D histogram data as a colored image.
    
    This widget displays a 2D numpy array as a colored image using a colormap.
    It supports:
    - Colormap application (via matplotlib)
    - Axis labels and scales
    - Mouse tracking for coordinate display
    - Background image display when no data is loaded
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self.setMouseTracking(True)
        
        # Data storage
        self._data: Optional[np.ndarray] = None
        self._colored_image: Optional[QImage] = None
        self._background_image: Optional[QImage] = None
        
        # Colormap settings
        self._colormap_name: str = "viridis"
        self._vmin: float = 0.0
        self._vmax: float = 1.0
        self._colormap = None
        self._norm = None
        
        # Axis settings
        self._x_min: float = 0.0
        self._x_max: float = 1.0
        self._y_min: float = 0.0
        self._y_max: float = 1.0
        self._show_axes: dict = {
            'bottom': False,
            'top': False,
            'left': False,
            'right': False
        }
        
        # Axis labels
        self._x_label: str = ""
        self._y_label: str = ""
        
        # Font settings
        self._axis_font = QFont("Courier", 8)
        
        # Background color
        self.setStyleSheet("background-color: white;")
        
        # Load matplotlib colormap
        self._load_colormap()
    
    def _load_colormap(self):
        """Load matplotlib colormap for data visualization."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import Normalize
            
            self._colormap = plt.get_cmap(self._colormap_name)
            self._norm = Normalize(vmin=self._vmin, vmax=self._vmax)
        except ImportError:
            logging.warning("matplotlib not available, using grayscale")
            self._colormap = None
            self._norm = None
        except Exception as e:
            logging.error(f"Failed to load colormap {self._colormap_name}: {e}")
            self._colormap = None
            self._norm = None
    
    def set_data(self, data: np.ndarray):
        """Set the 2D histogram data to display."""
        if data is None or data.size == 0:
            self._data = None
            self._colored_image = None
            self.update()
            return
        
        self._data = data
        self._update_colored_image()
        self.update()
    
    def set_background_image(self, image_path: str):
        """Set a background image to display when no data is loaded."""
        try:
            qimage = QImage(image_path)
            if not qimage.isNull():
                self._background_image = qimage
                self.update()
        except Exception as e:
            logging.warning(f"Failed to load background image: {e}")
    
    def set_colormap(self, colormap_name: str, vmin: Optional[float] = None, vmax: Optional[float] = None):
        """Set the colormap and value range."""
        self._colormap_name = colormap_name
        if vmin is not None:
            self._vmin = vmin
        if vmax is not None:
            self._vmax = vmax
        
        self._load_colormap()
        if self._data is not None:
            self._update_colored_image()
        self.update()
    
    def set_axis_scale(self, axis: str, min_val: float, max_val: float):
        """Set the scale for an axis (xBottom, yLeft, etc.)."""
        if axis in ('xBottom', 'x'):
            self._x_min = min_val
            self._x_max = max_val
        elif axis in ('yLeft', 'y'):
            self._y_min = min_val
            self._y_max = max_val
        self.update()
    
    def enable_axis(self, axis: str, enabled: bool):
        """Enable or disable axis display."""
        axis_map = {
            'xBottom': 'bottom',
            'xTop': 'top',
            'yLeft': 'left',
            'yRight': 'right'
        }
        if axis in axis_map:
            self._show_axes[axis_map[axis]] = enabled
            self.update()
    
    def axis_enabled(self, axis: str) -> bool:
        """Check if an axis is enabled."""
        axis_map = {
            'xBottom': 'bottom',
            'xTop': 'top',
            'yLeft': 'left',
            'yRight': 'right'
        }
        if axis in axis_map:
            return self._show_axes[axis_map[axis]]
        return False
    
    def set_axis_font(self, axis: str, font: QFont):
        """Set the font for axis labels."""
        self._axis_font = font
        self.update()
    
    def set_lut_range(self, range_values):
        """Set the LUT (lookup table) range for compatibility with guiqwt.
        
        This is a compatibility method that calls set_colormap with the current
        colormap name and the provided range values.
        
        Args:
            range_values: List or tuple of [vmin, vmax]
        """
        if len(range_values) >= 2:
            vmin, vmax = range_values[0], range_values[1]
            self.set_colormap(self._colormap_name, vmin, vmax)
    
    def _update_colored_image(self):
        """Apply colormap to data and create QImage."""
        if self._data is None:
            self._colored_image = None
            return
        
        try:
            # Apply colormap
            if self._colormap is not None and self._norm is not None:
                # Normalize and apply colormap
                normalized = self._norm(self._data)
                colored = self._colormap(normalized)
                
                # Convert to RGBA uint8
                rgba_uint8 = (colored * 255).astype(np.uint8)
                
                # Create QImage from RGBA data
                height, width = rgba_uint8.shape[:2]
                bytes_per_line = 4 * width
                
                # QImage expects RGBA format
                self._colored_image = QImage(
                    rgba_uint8.data,
                    width,
                    height,
                    bytes_per_line,
                    QImage.Format_RGBA8888
                ).copy()  # Copy to avoid data lifetime issues
            else:
                # Fallback: grayscale
                data_normalized = (self._data - self._data.min()) / (self._data.max() - self._data.min() + 1e-10)
                gray_uint8 = (data_normalized * 255).astype(np.uint8)
                height, width = gray_uint8.shape
                self._colored_image = QImage(
                    gray_uint8.data,
                    width,
                    height,
                    width,
                    QImage.Format_Grayscale8
                ).copy()
        except Exception as e:
            logging.error(f"Failed to create colored image: {e}")
            self._colored_image = None
    
    def paintEvent(self, event):
        """Paint the widget."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        
        # Calculate margins for axes
        margin_left = 50 if self._show_axes['left'] else 0
        margin_right = 50 if self._show_axes['right'] else 0
        margin_top = 30 if self._show_axes['top'] else 0
        margin_bottom = 30 if self._show_axes['bottom'] else 0
        
        # Image area
        image_rect = QRect(
            margin_left,
            margin_top,
            self.width() - margin_left - margin_right,
            self.height() - margin_top - margin_bottom
        )
        
        # Draw background
        painter.fillRect(self.rect(), Qt.white)
        
        # Draw image (data or background)
        if self._colored_image is not None and not self._colored_image.isNull():
            # Draw the colored histogram data
            # Mirror vertically to match guiqwt orientation
            scaled_image = self._colored_image.scaled(
                image_rect.size(),
                Qt.IgnoreAspectRatio,
                Qt.FastTransformation
            ).mirrored(False, True)  # Flip vertically
            painter.drawImage(image_rect.topLeft(), scaled_image)
        elif self._background_image is not None and not self._background_image.isNull():
            # Draw background image (centered, aspect ratio preserved)
            scaled_bg = self._background_image.scaled(
                image_rect.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            x_offset = (image_rect.width() - scaled_bg.width()) // 2
            y_offset = (image_rect.height() - scaled_bg.height()) // 2
            painter.drawImage(
                image_rect.x() + x_offset,
                image_rect.y() + y_offset,
                scaled_bg
            )
        
        # Draw axes if enabled
        self._draw_axes(painter, image_rect)
        
        painter.end()
    
    def _draw_axes(self, painter: QPainter, image_rect: QRect):
        """Draw axis labels and ticks."""
        painter.setFont(self._axis_font)
        painter.setPen(QPen(Qt.black, 1))
        
        # Bottom axis
        if self._show_axes['bottom']:
            y_pos = image_rect.bottom() + 20
            painter.drawText(
                QRect(image_rect.left(), y_pos, image_rect.width(), 20),
                Qt.AlignCenter,
                f"{self._x_min:.1f} - {self._x_max:.1f}"
            )
        
        # Top axis
        if self._show_axes['top']:
            y_pos = image_rect.top() - 20
            painter.drawText(
                QRect(image_rect.left(), y_pos, image_rect.width(), 20),
                Qt.AlignCenter,
                f"{self._x_min:.1f} - {self._x_max:.1f}"
            )
        
        # Left axis
        if self._show_axes['left']:
            painter.save()
            painter.translate(10, image_rect.center().y())
            painter.rotate(-90)
            painter.drawText(
                QRect(-100, -10, 200, 20),
                Qt.AlignCenter,
                f"{self._y_min:.1f} - {self._y_max:.1f}"
            )
            painter.restore()
        
        # Right axis
        if self._show_axes['right']:
            painter.save()
            painter.translate(self.width() - 10, image_rect.center().y())
            painter.rotate(-90)
            painter.drawText(
                QRect(-100, -10, 200, 20),
                Qt.AlignCenter,
                f"{self._y_min:.1f} - {self._y_max:.1f}"
            )
            painter.restore()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events for coordinate tracking."""
        super().mouseMoveEvent(event)
    
    def canvas(self):
        """Return self for compatibility with guiqwt API."""
        return self
    
    def replot(self):
        """Trigger a repaint (compatibility with guiqwt)."""
        self.update()
    
    def invTransform(self, axis_id, pixel_pos):
        """Convert pixel position to data coordinates (compatibility with Qwt).
        
        Args:
            axis_id: Axis identifier (0=xBottom, 1=yLeft)
            pixel_pos: Pixel position
            
        Returns:
            Data coordinate value
        """
        # Calculate margins for axes
        margin_left = 50 if self._show_axes['left'] else 0
        margin_right = 50 if self._show_axes['right'] else 0
        margin_top = 30 if self._show_axes['top'] else 0
        margin_bottom = 30 if self._show_axes['bottom'] else 0
        
        # Image area dimensions
        image_width = self.width() - margin_left - margin_right
        image_height = self.height() - margin_top - margin_bottom
        
        if axis_id == 0:  # xBottom
            relative_pos = pixel_pos - margin_left
            if image_width > 0:
                fraction = relative_pos / image_width
                return fraction * (self._x_max - self._x_min) + self._x_min
        else:  # yLeft
            relative_pos = pixel_pos - margin_top
            if image_height > 0:
                # Y-axis is inverted after vertical flip
                fraction = 1.0 - (relative_pos / image_height)
                return fraction * (self._y_max - self._y_min) + self._y_min
        
        return 0.0
    
    @property
    def xBottom(self):
        """Return x-axis identifier for compatibility."""
        return 0
    
    @property  
    def yLeft(self):
        """Return y-axis identifier for compatibility."""
        return 1
    
    def canvasMap(self, axis_id):
        """Return a map object for coordinate transformation (compatibility with Qwt).
        
        This is a simplified version that returns an object with transform/invTransform methods.
        """
        class AxisMap:
            def __init__(self, widget, is_x_axis):
                self.widget = widget
                self.is_x_axis = is_x_axis
                
            def transform(self, data_value):
                """Convert data coordinate to pixel position."""
                margin_left = 50 if self.widget._show_axes['left'] else 0
                margin_right = 50 if self.widget._show_axes['right'] else 0
                margin_top = 30 if self.widget._show_axes['top'] else 0
                margin_bottom = 30 if self.widget._show_axes['bottom'] else 0
                
                image_width = self.widget.width() - margin_left - margin_right
                image_height = self.widget.height() - margin_top - margin_bottom
                
                if self.is_x_axis:
                    fraction = (data_value - self.widget._x_min) / (self.widget._x_max - self.widget._x_min)
                    return margin_left + fraction * image_width
                else:
                    # Y-axis is inverted after vertical flip
                    fraction = (data_value - self.widget._y_min) / (self.widget._y_max - self.widget._y_min)
                    return margin_top + (1.0 - fraction) * image_height
            
            def invTransform(self, pixel_pos):
                """Convert pixel position to data coordinate."""
                margin_left = 50 if self.widget._show_axes['left'] else 0
                margin_right = 50 if self.widget._show_axes['right'] else 0
                margin_top = 30 if self.widget._show_axes['top'] else 0
                margin_bottom = 30 if self.widget._show_axes['bottom'] else 0
                
                image_width = self.widget.width() - margin_left - margin_right
                image_height = self.widget.height() - margin_top - margin_bottom
                
                if self.is_x_axis:
                    relative_pos = pixel_pos - margin_left
                    if image_width > 0:
                        fraction = relative_pos / image_width
                        return fraction * (self.widget._x_max - self.widget._x_min) + self.widget._x_min
                else:
                    relative_pos = pixel_pos - margin_top
                    if image_height > 0:
                        # Y-axis is inverted after vertical flip
                        fraction = 1.0 - (relative_pos / image_height)
                        return fraction * (self.widget._y_max - self.widget._y_min) + self.widget._y_min
                
                return 0.0
        
        return AxisMap(self, axis_id == 0)
