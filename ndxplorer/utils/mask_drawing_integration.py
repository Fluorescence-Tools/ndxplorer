"""
Integration of mask drawing with NDXplorer 2D plot.

Handles mouse events for drawing masks on the 2D histogram.
"""
from typing import Optional, Tuple
import numpy as np

from qtpy import QtCore, QtGui, QtWidgets

from . import mask_helpers
from ..widgets.mask_overlay_widget import MaskOverlayWidget


class MaskDrawingIntegration(QtCore.QObject):
    """
    Integrates mask drawing with NDXplorer's 2D plot.
    
    Handles:
    - Mouse events for drawing
    - Mask overlay visualization
    - Coordinate transformation between plot and image space
    """
    
    def __init__(self, parent):
        """
        Initialize mask drawing integration.
        
        Parameters
        ----------
        parent : NDXplorer
            The parent NDXplorer instance
        """
        super().__init__(parent)
        self.parent = parent
        self._mask_overlay_widget = None
        self._is_drawing = False
        self._last_draw_pos = None
        
    def setup_mask_overlay(self):
        """Setup the mask overlay widget on the 2D plot."""
        try:
            # Get the 2D plot canvas
            canvas = self.parent.g_2dplot.canvas()
            if canvas is None:
                return
            
            # Create overlay widget as a child of the canvas
            self._mask_overlay_widget = MaskOverlayWidget(canvas)
            self._mask_overlay_widget.setGeometry(canvas.rect())
            self._mask_overlay_widget.show()
            
            # Ensure overlay stays on top
            self._mask_overlay_widget.raise_()
            
            # Sync brush size from mask widget
            mask_widget = self.parent.plot_control.mask_widget
            self._mask_overlay_widget.set_brush_radius(mask_widget._brush_size)
            self._update_brush_pixel_scale()
            
            # Connect brush size changes
            mask_widget.brush_spinbox.valueChanged.connect(
                lambda size: self._mask_overlay_widget.set_brush_radius(size)
            )

            mask_widget.brush_spinbox.valueChanged.connect(
                lambda _: self._update_brush_pixel_scale()
            )
            
            if canvas is not None:
                # Connect to canvas resize events
                canvas.installEventFilter(self)
            
            from ..logging_config import logging
            logging.debug("Mask overlay widget created")
        
            
        except Exception as e:
            from ..logging_config import logging
            logging.warning(f"Could not setup mask overlay: {e}")

    def _update_brush_pixel_scale(self):
        if self._mask_overlay_widget is None:
            return

        mask_widget = self.parent.plot_control.mask_widget
        mask = mask_widget.get_mask()
        if mask is None:
            return

        try:
            plot = self.parent.g_2dplot
            xMap = plot.canvasMap(plot.xBottom)
            yMap = plot.canvasMap(plot.yLeft)

            ny, nx = mask.shape
            if nx <= 1 or ny <= 1:
                return

            x0 = xMap.transform(0)
            x1 = xMap.transform(1)
            y0 = yMap.transform(0)
            y1 = yMap.transform(1)

            px_per_bin_x = float(abs(x1 - x0))
            px_per_bin_y = float(abs(y1 - y0))
            if px_per_bin_x <= 0 or px_per_bin_y <= 0:
                return

            mask_widget.set_brush_pixel_scale(px_per_bin_x, px_per_bin_y)
        except Exception:
            return
    
    def update_mask_overlay(self, mask: Optional[np.ndarray] = None):
        """
        Update the mask overlay visualization.
        
        Parameters
        ----------
        mask : Optional[np.ndarray]
            The mask to display. If None, uses the mask from the widget.
        """
        from ..logging_config import logging
        
        if self._mask_overlay_widget is None:
            logging.info("Mask overlay widget is None, setting up...")
            self.setup_mask_overlay()
        
        if self._mask_overlay_widget is None:
            logging.warning("Could not create mask overlay widget")
            return
        
        if mask is None:
            mask = self.parent.plot_control.mask_widget.get_mask()
        
        if mask is None:
            self._mask_overlay_widget.clear_mask()
            return
        
        try:
            # We don't need bounds anymore since we use QwtScaleMap in the widget
            self._mask_overlay_widget.set_mask(mask)
            self._update_brush_pixel_scale()
            
        except Exception as e:
            from ..logging_config import logging
            logging.warning(f"Could not update mask overlay: {e}")
    
    def _mask_to_rgba(self, mask: np.ndarray) -> np.ndarray:
        """
        Convert a mask to an RGBA overlay image.
        
        Parameters
        ----------
        mask : np.ndarray
            Integer mask with class labels
            
        Returns
        -------
        rgba : np.ndarray
            RGBA image with shape (H, W, 4)
        """
        # Create RGBA image - ensure dimensions are integers
        h, w = int(mask.shape[0]), int(mask.shape[1])
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Define colors for different classes
        colors = [
            (255, 0, 0, 100),      # Class 1: Red
            (0, 255, 0, 100),      # Class 2: Green
            (0, 0, 255, 100),      # Class 3: Blue
            (255, 255, 0, 100),    # Class 4: Yellow
            (255, 0, 255, 100),    # Class 5: Magenta
            (0, 255, 255, 100),    # Class 6: Cyan
            (255, 128, 0, 100),    # Class 7: Orange
            (128, 0, 255, 100),    # Class 8: Purple
        ]
        
        # Apply colors based on class
        for class_id in range(1, 256):
            if not np.any(mask == class_id):
                continue
            
            color = colors[(class_id - 1) % len(colors)]
            mask_pixels = mask == class_id
            rgba[mask_pixels] = color
        
        return rgba
    
    def handle_mouse_press(self, event):
        """
        Handle mouse press event for mask drawing.
        
        Parameters
        ----------
        event : QMouseEvent
            The mouse event
        """
        from ..logging_config import logging
        if event.button() == QtCore.Qt.LeftButton:
            logging.info("Left button pressed")
            self._is_drawing = True
            pos = self._get_image_coordinates(event)
            logging.info(f"Image coordinates: {pos}")
            if pos is not None:
                self._draw_at_position(pos)
                self._last_draw_pos = pos
            else:
                logging.warning("Could not get image coordinates")
    
    def handle_mouse_move(self, event):
        """
        Handle mouse move event for mask drawing.
        
        Parameters
        ----------
        event : QMouseEvent
            The mouse event
        """
        if self._is_drawing:
            pos = self._get_image_coordinates(event)
            if pos is not None:
                # Draw line from last position to current position
                if self._last_draw_pos is not None:
                    self._draw_line(self._last_draw_pos, pos)
                else:
                    self._draw_at_position(pos)
                self._last_draw_pos = pos
    
    def handle_mouse_release(self, event):
        """
        Handle mouse release event for mask drawing.
        
        Parameters
        ----------
        event : QMouseEvent
            The mouse event
        """
        if event.button() == QtCore.Qt.LeftButton:
            self._is_drawing = False
            self._last_draw_pos = None
            # Emit mask_changed signal now to trigger histogram update
            mask_widget = self.parent.plot_control.mask_widget
            mask_widget.emit_mask_changed()
    
    def _get_image_coordinates(self, event) -> Optional[Tuple[int, int]]:
        """
        Convert mouse event coordinates to image indices.
        
        Returns
        -------
        pos : Optional[Tuple[int, int]]
            (row, col) indices in the mask array (ny, nx)
        """
        try:
            plot = self.parent.g_2dplot
            
            # Use Qwt's inverse transform to get plot coordinates
            # Note: We mapped the axis scale to [0, n_bins] in plot_main.py
            # self.g_2dplot.setAxisScale(QwtPlot.xBottom, 0, len(x_edges)-1)
            # self.g_2dplot.setAxisScale(QwtPlot.yLeft, 0, len(y_edges)-1)
            
            pos = event.pos()
            x_idx = int(plot.invTransform(plot.xBottom, pos.x()))
            y_idx = int(plot.invTransform(plot.yLeft, pos.y()))
            
            mask = self.parent.plot_control.mask_widget.get_mask()
            if mask is None:
                return None
            
            ny, nx = mask.shape
            
            # Check bounds
            if 0 <= x_idx < nx and 0 <= y_idx < ny:
                return (y_idx, x_idx)
            
        except Exception as e:
            from ..logging_config import logging
            logging.debug(f"Error getting image coordinates: {e}")
        
        return None
    
    def _draw_at_position(self, pos: Tuple[int, int]):
        """
        Draw at the specified position.
        
        Parameters
        ----------
        pos : Tuple[int, int]
            Position (y, x) in image coordinates
        """
        from ..logging_config import logging
        mask_widget = self.parent.plot_control.mask_widget
        logging.info(f"Drawing at position {pos}")
        # Use silent=True to suppress mask_changed signal during drawing
        mask_widget.apply_brush(pos, silent=True)
        logging.info("Brush applied, updating overlay")
        # Force immediate overlay update
        self.update_mask_overlay()
        logging.info("Overlay updated")
    
    def _draw_line(self, pos1: Tuple[int, int], pos2: Tuple[int, int]):
        """
        Draw a line between two positions.
        
        Parameters
        ----------
        pos1 : Tuple[int, int]
            Start position (y, x)
        pos2 : Tuple[int, int]
            End position (y, x)
        """
        # Use Bresenham's line algorithm to draw between points
        y1, x1 = pos1
        y2, x2 = pos2
        
        points = self._bresenham_line(x1, y1, x2, y2)
        
        mask_widget = self.parent.plot_control.mask_widget
        # Use silent=True to suppress mask_changed signal during drawing
        for y, x in points:
            mask_widget.apply_brush((y, x), silent=True)
    
    def _bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> list:
        """
        Generate points along a line using Bresenham's algorithm.
        
        Parameters
        ----------
        x0, y0 : int
            Start point
        x1, y1 : int
            End point
            
        Returns
        -------
        points : list
            List of (y, x) tuples along the line
        """
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            points.append((y, x))
            
            if x == x1 and y == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return points
    
    def enable_drawing_mode(self):
        """Enable mask drawing mode on the 2D plot."""
        from ..logging_config import logging
        try:
            logging.info("enable_drawing_mode called")
            
            # Check if overlay_plot exists
            if not hasattr(self.parent, 'overlay_plot'):
                logging.error("Parent does not have overlay_plot attribute")
                return
            
            logging.info(f"overlay_plot exists: {self.parent.overlay_plot}")
            
            overlay_widget = self.parent.overlay_plot
            if overlay_widget is None:
                logging.error("overlay_widget is None")
                return

            if hasattr(self.parent, 'mouse_event_filter') and self.parent.mouse_event_filter is not None:
                overlay_widget.removeEventFilter(self.parent.mouse_event_filter)

            overlay_widget.installEventFilter(self)
            logging.info("Event filter installed on overlay widget")
            
            # Ensure cursor overlay is visible when drawing mode is enabled
            if self._mask_overlay_widget is not None:
                self._mask_overlay_widget.set_cursor_visible(True)
                logging.info("Cursor overlay made visible")
            
            logging.info("Mask drawing mode enabled (rectangular selection disabled)")
        except Exception as e:
            logging.error(f"Could not enable drawing mode: {e}", exc_info=True)
    
    def disable_drawing_mode(self):
        """Disable mask drawing mode on the 2D plot."""
        try:
            # Hide cursor overlay when drawing mode is disabled
            if self._mask_overlay_widget is not None:
                self._mask_overlay_widget.set_cursor_visible(False)
                self._mask_overlay_widget.set_cursor_pos(None)
                # Hide the mask overlay when drawing mode is disabled
                self._mask_overlay_widget.clear_mask()
            
            if hasattr(self.parent, 'overlay_plot'):
                overlay_widget = self.parent.overlay_plot
                if overlay_widget is not None:
                    overlay_widget.removeEventFilter(self)
                    if hasattr(self.parent, 'mouse_event_filter') and self.parent.mouse_event_filter is not None:
                        overlay_widget.installEventFilter(self.parent.mouse_event_filter)
            
            from ..logging_config import logging
            logging.info("Mask drawing mode disabled (rectangular selection enabled, draw overlays hidden)")
        except Exception as e:
            from ..logging_config import logging
            logging.warning(f"Could not disable drawing mode: {e}")
    
    def eventFilter(self, obj, event):
        """
        Event filter for capturing mouse events on the plot canvas.
        
        Parameters
        ----------
        obj : QObject
            The object that received the event
        event : QEvent
            The event
            
        Returns
        -------
        bool
            True if event was handled, False otherwise
        """
        from ..logging_config import logging
        
        # Safety check: ensure parent is valid and has required attributes
        if not hasattr(self, 'parent') or not hasattr(self.parent, 'plot_control'):
            return False
        
        # Handle canvas resize events to keep overlay positioned correctly
        if event.type() == QtCore.QEvent.Resize:
            if self._mask_overlay_widget is not None:
                if hasattr(self.parent, 'g_2dplot'):
                    canvas = self.parent.g_2dplot.canvas()
                    if canvas is not None:
                        self._mask_overlay_widget.setGeometry(canvas.rect())
                        self._update_brush_pixel_scale()
            return False

        if event.type() == QtCore.QEvent.Enter:
            if self._mask_overlay_widget is not None:
                drawing_enabled = self.parent.plot_control.mask_widget.is_drawing_enabled()
                self._mask_overlay_widget.set_cursor_visible(bool(drawing_enabled))
            return False

        if event.type() == QtCore.QEvent.Leave:
            if self._mask_overlay_widget is not None:
                self._mask_overlay_widget.set_cursor_visible(False)
                self._mask_overlay_widget.set_cursor_pos(None)
            return False
        
        # Check if drawing is enabled
        if not hasattr(self.parent.plot_control, 'mask_widget'):
            return False
        drawing_enabled = self.parent.plot_control.mask_widget.is_drawing_enabled()
        
        # Handle mouse events for drawing
        if event.type() == QtCore.QEvent.MouseButtonPress:
            logging.info(f"Mouse press event, drawing_enabled={drawing_enabled}")
            if drawing_enabled:
                self.handle_mouse_press(event)
                return True  # Consume the event to prevent rectangular selection
            return False
        if event.type() == QtCore.QEvent.MouseMove:
            if self._mask_overlay_widget is not None:
                try:
                    g_canvas = self.parent.g_2dplot.canvas()
                    overlay_canvas = obj
                    global_pos = overlay_canvas.mapToGlobal(event.pos())
                    pos_on_g = g_canvas.mapFromGlobal(global_pos)
                    
                    # Always show cursor when drawing is enabled, regardless of whether we're actively drawing
                    drawing_enabled = self.parent.plot_control.mask_widget.is_drawing_enabled()
                    self._mask_overlay_widget.set_cursor_visible(bool(drawing_enabled))
                    self._mask_overlay_widget.set_cursor_pos(pos_on_g)
                    
                    # Debug logging to help track cursor behavior
                    if drawing_enabled:
                        from ..logging_config import logging
                        logging.debug(f"Cursor visible at pos: {pos_on_g.x()}, {pos_on_g.y()}")
                        
                except Exception as e:
                    from ..logging_config import logging
                    logging.debug(f"Error updating cursor position: {e}")

            if drawing_enabled and self._is_drawing:
                self.handle_mouse_move(event)
                return True  # Consume the event while drawing
            return False
        elif event.type() == QtCore.QEvent.MouseButtonRelease:
            if drawing_enabled and self._is_drawing:
                self.handle_mouse_release(event)
                return True  # Consume the event
            return False
        
        return False
