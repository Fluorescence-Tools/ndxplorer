"""
PyQt-based overlay widget for drawing selections and curves on the 2D histogram.
Replaces the guiqwt overlay plot.
"""

from typing import List, Tuple, Optional
import numpy as np
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt, QRect, QPoint, QPointF
from qtpy.QtGui import QPainter, QPen, QColor, QBrush

from ..logging_config import logging


class DrawingOverlayWidget(QtWidgets.QWidget):
    """
    Transparent overlay widget for drawing selections and curves on top of the 2D histogram.
    
    This replaces the guiqwt overlay plot with a pure PyQt implementation.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Make widget transparent
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)
        
        # Drawing state
        self._rectangles: List[Tuple[QRect, QColor]] = []
        self._curves: List[Tuple[List[QPointF], QColor, int]] = []  # points, color, width
        self._mask_overlay_data: Optional[np.ndarray] = None
        self._mask_colormap = None
        
        # Prevent recursive repaints
        self._painting = False
        
        # Axis scales for coordinate transformation
        self._x_min = 0.0
        self._x_max = 1.0
        self._y_min = 0.0
        self._y_max = 1.0
        
        # Margins (should match SimpleImageWidget)
        self._margin_left = 0
        self._margin_right = 0
        self._margin_top = 0
        self._margin_bottom = 0
    
    def set_axis_scale(self, axis: str, min_val: float, max_val: float):
        """Set the axis scale for coordinate transformation."""
        if axis in ('xBottom', 'x'):
            self._x_min = min_val
            self._x_max = max_val
        elif axis in ('yLeft', 'y'):
            self._y_min = min_val
            self._y_max = max_val
        self.update()
    
    def set_margins(self, left: int, right: int, top: int, bottom: int):
        """Set the margins to match the underlying plot."""
        self._margin_left = left
        self._margin_right = right
        self._margin_top = top
        self._margin_bottom = bottom
        self.update()
    
    def add_rectangle(self, x1: float, y1: float, x2: float, y2: float, color: QColor = None):
        """Add a rectangle in data coordinates."""
        if color is None:
            color = QColor(255, 0, 0, 128)  # Semi-transparent red
        
        # Convert data coordinates to widget coordinates
        px1 = self._data_to_pixel_x(x1)
        py1 = self._data_to_pixel_y(y1)
        px2 = self._data_to_pixel_x(x2)
        py2 = self._data_to_pixel_y(y2)
        
        rect = QRect(
            int(min(px1, px2)),
            int(min(py1, py2)),
            int(abs(px2 - px1)),
            int(abs(py2 - py1))
        )
        self._rectangles.append((rect, color))
        self.update()
    
    def add_curve(self, x_data: np.ndarray, y_data: np.ndarray, color: QColor = None, width: int = 2):
        """Add a curve in data coordinates."""
        if color is None:
            color = QColor(255, 255, 0, 255)  # Yellow
        
        # Convert data coordinates to widget coordinates
        points = []
        for x, y in zip(x_data, y_data):
            px = self._data_to_pixel_x(x)
            py = self._data_to_pixel_y(y)
            points.append(QPointF(px, py))
        
        self._curves.append((points, color, width))
        self.update()
    
    def clear_rectangles(self):
        """Clear all rectangles."""
        self._rectangles.clear()
        self.update()
    
    def clear_curves(self):
        """Clear all curves."""
        self._curves.clear()
        self.update()
    
    def clear_all(self):
        """Clear all overlay items."""
        self._rectangles.clear()
        self._curves.clear()
        self._mask_overlay_data = None
        self.update()
    
    def set_mask_overlay(self, mask_data: np.ndarray, colormap=None):
        """Set mask overlay data for visualization."""
        self._mask_overlay_data = mask_data
        self._mask_colormap = colormap
        self.update()
    
    def _data_to_pixel_x(self, x: float) -> float:
        """Convert data x-coordinate to pixel coordinate."""
        image_width = self.width() - self._margin_left - self._margin_right
        if self._x_max - self._x_min == 0:
            return self._margin_left
        fraction = (x - self._x_min) / (self._x_max - self._x_min)
        return self._margin_left + fraction * image_width
    
    def _data_to_pixel_y(self, y: float) -> float:
        """Convert data y-coordinate to pixel coordinate."""
        image_height = self.height() - self._margin_top - self._margin_bottom
        if self._y_max - self._y_min == 0:
            return self._margin_top
        # Y-axis is inverted (after vertical flip)
        fraction = (y - self._y_min) / (self._y_max - self._y_min)
        return self._margin_top + (1.0 - fraction) * image_height
    
    def paintEvent(self, event):
        """Paint the overlay items."""
        # Prevent recursive repaints
        if self._painting:
            return
        
        self._painting = True
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing, True)
            
            # Draw mask overlay if present
            if self._mask_overlay_data is not None:
                self._draw_mask_overlay(painter)
            
            # Draw rectangles
            for rect, color in self._rectangles:
                pen = QPen(color, 2)
                painter.setPen(pen)
                brush = QBrush(QColor(color.red(), color.green(), color.blue(), 64))
                painter.setBrush(brush)
                painter.drawRect(rect)
            
            # Draw curves
            for points, color, width in self._curves:
                if len(points) < 2:
                    continue
                
                # Convert color to QColor if it's a string
                if isinstance(color, str):
                    qcolor = QColor(color)
                else:
                    qcolor = color
                
                pen = QPen(qcolor, width)
                pen.setCapStyle(Qt.RoundCap)
                pen.setJoinStyle(Qt.RoundJoin)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                
                # Draw line segments
                for i in range(len(points) - 1):
                    painter.drawLine(points[i], points[i + 1])
            
            painter.end()
        finally:
            self._painting = False
    
    def _draw_mask_overlay(self, painter: QPainter):
        """Draw the mask overlay."""
        # This would render the mask data as a semi-transparent overlay
        # For now, we'll skip this as it's complex and the mask is handled elsewhere
        pass
    
    def replot(self):
        """Trigger a repaint (for compatibility with guiqwt API)."""
        self.update()
    
    def del_item(self, item):
        """Remove an item (for compatibility with guiqwt API)."""
        # This is a placeholder for compatibility
        pass
    
    def add_item(self, item):
        """Add an item (for compatibility with guiqwt API)."""
        # This is a placeholder for compatibility
        pass
    
    def setAxisScale(self, axis_id, min_val: float, max_val: float):
        """Set axis scale (for compatibility with Qwt API)."""
        if axis_id == 0:  # xBottom
            self.set_axis_scale('xBottom', min_val, max_val)
        elif axis_id == 1:  # yLeft
            self.set_axis_scale('yLeft', min_val, max_val)
    
    def canvas(self):
        """Return self for compatibility."""
        return self
    
    def enableAxis(self, axis_id, enabled: bool):
        """Enable/disable axis (for compatibility, does nothing)."""
        pass
