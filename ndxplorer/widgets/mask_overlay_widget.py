"""
Transparent overlay widget for drawing masks on the 2D plot.
"""
from typing import Optional
import numpy as np

try:
    from chisurf.gui import QtCore, QtGui, QtWidgets
except ImportError:
    from qtpy import QtCore, QtGui, QtWidgets


class MaskOverlayWidget(QtWidgets.QWidget):
    """
    Transparent widget that overlays the 2D plot for mask visualization.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Make widget transparent
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        
        # Enable mouse tracking for cursor preview
        self.setMouseTracking(True)
        
        # Store mask data
        self._mask = None
        self._xedges = None
        self._yedges = None
        
        # Cursor preview
        self._cursor_pos = None
        self._brush_radius = 5
        self._show_cursor = False
        
        # Color map for different classes
        self._colors = [
            QtGui.QColor(255, 0, 0, 100),      # Class 1: Red
            QtGui.QColor(0, 255, 0, 100),      # Class 2: Green
            QtGui.QColor(0, 0, 255, 100),      # Class 3: Blue
            QtGui.QColor(255, 255, 0, 100),    # Class 4: Yellow
            QtGui.QColor(255, 0, 255, 100),    # Class 5: Magenta
            QtGui.QColor(0, 255, 255, 100),    # Class 6: Cyan
            QtGui.QColor(255, 128, 0, 100),    # Class 7: Orange
            QtGui.QColor(128, 0, 255, 100),    # Class 8: Purple
        ]
    
    def set_mask(self, mask: Optional[np.ndarray], xedges: Optional[np.ndarray] = None, yedges: Optional[np.ndarray] = None):
        """
        Set the mask to display.
        
        Parameters
        ----------
        mask : Optional[np.ndarray]
            Integer mask with class labels (H, W) or (ny, nx)
        xedges : Optional[np.ndarray]
            X bin edges
        yedges : Optional[np.ndarray]
            Y bin edges
        """
        self._mask = mask
        self._xedges = xedges
        self._yedges = yedges
        self.update()
    
    def clear_mask(self):
        """Clear the mask overlay."""
        self._mask = None
        self._xedges = None
        self._yedges = None
        self.update()
    
    def set_brush_radius(self, radius: int):
        """Set the brush radius for cursor preview."""
        self._brush_radius = radius
        self.update()
    
    def set_cursor_visible(self, visible: bool):
        """Set whether the cursor preview is visible."""
        self._show_cursor = visible
        self.update()

    def set_cursor_pos(self, pos: Optional[QtCore.QPoint]):
        self._cursor_pos = pos
        self.update()
    
    def mouseMoveEvent(self, event):
        """Track mouse position for cursor preview."""
        self._cursor_pos = event.pos()
        self.update()
        super().mouseMoveEvent(event)
    
    def enterEvent(self, event):
        """Show cursor when mouse enters widget."""
        self._show_cursor = True
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        """Hide cursor when mouse leaves widget."""
        self._show_cursor = False
        self._cursor_pos = None
        self.update()
        super().leaveEvent(event)
    
    def paintEvent(self, event):
        """Paint the mask overlay and cursor."""
        painter = QtGui.QPainter(self)
        
        try:
            # Get the plot and scale maps for perfect alignment
            # The parent is the canvas
            canvas = self.parent()
            
            # Check if using SimpleImageWidget or Qwt plot
            if hasattr(canvas, 'canvasMap'):
                # SimpleImageWidget or compatible widget
                xMap = canvas.canvasMap(0)  # xBottom
                yMap = canvas.canvasMap(1)  # yLeft
            elif hasattr(canvas, 'plot'):
                # Qwt canvas - get the plot first
                plot = canvas.plot()
                xMap = plot.canvasMap(plot.xBottom)
                yMap = plot.canvasMap(plot.yLeft)
            else:
                # Can't get coordinate mapping, skip drawing
                return
            
            # Draw mask overlay
            if self._mask is not None:
                painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
                
                ny, nx = self._mask.shape
                
                # Iterate over non-zero pixels in the mask
                # This is more efficient than a full loop for sparse masks
                y_indices, x_indices = np.where(self._mask > 0)
                
                for iy, ix in zip(y_indices, x_indices):
                    class_id = self._mask[iy, ix]
                    
                    # Get pixel bounds in plot coordinates (bin indices)
                    # Data at (iy, ix) covers [ix, ix+1] on x-axis and [iy, iy+1] on y-axis
                    # Since the image is transposed H.T, H[ix, iy] is at (ix, iy)
                    
                    # Convert data coordinates (bin indices) to pixel coordinates
                    x1 = xMap.transform(ix)
                    x2 = xMap.transform(ix + 1)
                    y1 = yMap.transform(iy)
                    y2 = yMap.transform(iy + 1)
                    
                    # Rect for this bin
                    rect = QtCore.QRectF(
                        min(x1, x2),
                        min(y1, y2),
                        abs(x2 - x1),
                        abs(y1 - y2)
                    )
                    
                    # Get color for this class
                    color = self._colors[(class_id - 1) % len(self._colors)]
                    painter.fillRect(rect, color)
            
            # Draw cursor circle
            if self._show_cursor and self._cursor_pos is not None:
                painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
                
                # Draw circle outline
                pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 200))
                pen.setWidth(2)
                painter.setPen(pen)
                painter.setBrush(QtCore.Qt.NoBrush)
                
                # Draw outer circle (brush size)
                painter.drawEllipse(
                    self._cursor_pos,
                    self._brush_radius,  # Use radius directly, not *2
                    self._brush_radius
                )
                
                # Draw center crosshair
                pen.setWidth(1)
                painter.setPen(pen)
                crosshair_size = 3
                painter.drawLine(
                    self._cursor_pos.x() - crosshair_size,
                    self._cursor_pos.y(),
                    self._cursor_pos.x() + crosshair_size,
                    self._cursor_pos.y()
                )
                painter.drawLine(
                    self._cursor_pos.x(),
                    self._cursor_pos.y() - crosshair_size,
                    self._cursor_pos.x(),
                    self._cursor_pos.y() + crosshair_size
                )
        
        except Exception as e:
            # Silently fail to avoid breaking the UI
            pass
        
        finally:
            painter.end()
    
    def get_plot_coordinates(self, pos: QtCore.QPoint) -> Optional[tuple]:
        """
        Convert widget coordinates to plot coordinates.
        
        Parameters
        ----------
        pos : QtCore.QPoint
            Position in widget coordinates
            
        Returns
        -------
        coords : Optional[tuple]
            (x, y) in plot coordinates, or None if invalid
        """
        if self._mask_bounds is None:
            return None
        
        widget_width = self.width()
        widget_height = self.height()
        
        if widget_width <= 0 or widget_height <= 0:
            return None
        
        xmin, xmax, ymin, ymax = self._mask_bounds
        
        # Convert to plot coordinates
        x = xmin + (pos.x() / widget_width) * (xmax - xmin)
        y = ymin + (pos.y() / widget_height) * (ymax - ymin)
        
        return (x, y)
    
    def get_mask_indices(self, pos: QtCore.QPoint) -> Optional[tuple]:
        """
        Convert widget coordinates to mask array indices.
        
        Parameters
        ----------
        pos : QtCore.QPoint
            Position in widget coordinates
            
        Returns
        -------
        indices : Optional[tuple]
            (row, col) indices in mask array, or None if invalid
        """
        if self._mask is None:
            return None
        
        widget_width = self.width()
        widget_height = self.height()
        
        if widget_width <= 0 or widget_height <= 0:
            return None
        
        mask_height, mask_width = self._mask.shape
        
        # Convert to mask indices
        col = int(pos.x() * mask_width / widget_width)
        row = int(pos.y() * mask_height / widget_height)
        
        # Check bounds
        if 0 <= row < mask_height and 0 <= col < mask_width:
            return (row, col)
        
        return None
