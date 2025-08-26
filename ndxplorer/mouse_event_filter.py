"""
Mouse event filter for handling region selection in 2D plots.
"""

from qtpy import QtCore, QtWidgets

class MouseEventFilter(QtCore.QObject):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.selecting = False
        self.start_pos = None
        self.current_pos = None
        self.selection_rect = None
        # Mode: 'rectangle' for region selection (default), 'point' for single-click selection
        self.mode = 'rectangle'
        # Optional callback for point selection mode
        self.point_callback = None

    def set_point_mode(self, enabled: bool, callback=None):
        """
        Enable/disable point selection mode. When enabled, a left click will invoke the
        provided callback with the click position as argument and will not start a rubber band.
        """
        self.mode = 'point' if enabled else 'rectangle'
        self.point_callback = callback if enabled else None

    def eventFilter(self, obj, event):
        # Allow right-click events for context menu
        if event.type() == QtCore.QEvent.MouseButtonPress and event.button() == QtCore.Qt.RightButton:
            return False  # Process right-click events normally

        # Handle left-click events
        if event.type() == QtCore.QEvent.MouseButtonPress and event.button() == QtCore.Qt.LeftButton:
            if self.mode == 'point':
                # Call the point selection callback if available
                if callable(self.point_callback):
                    try:
                        self.point_callback(event.pos())
                    except Exception:
                        pass
                return True  # Consume the event
            else:
                # Rectangle selection mode
                self.start_selection(event.pos())
                return True

        # Handle mouse move events for updating the selection rectangle
        if event.type() == QtCore.QEvent.MouseMove and self.selecting and self.mode == 'rectangle':
            self.update_selection(event.pos())
            return True

        # Handle left-button release events for finalizing the selection
        if event.type() == QtCore.QEvent.MouseButtonRelease and event.button() == QtCore.Qt.LeftButton and self.selecting and self.mode == 'rectangle':
            self.finish_selection(event.pos())
            return True

        # Ignore other mouse events that would trigger zoom and panning
        if event.type() in [QtCore.QEvent.MouseButtonDblClick, QtCore.QEvent.Wheel]:
            return True  # Ignore mouse events

        return False  # Process other events normally

    def start_selection(self, pos):
        self.selecting = True
        self.start_pos = pos
        self.current_pos = pos

        # Create a selection rectangle if it doesn't exist
        if self.selection_rect is None:
            self.selection_rect = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self.parent.overlay_plot.canvas())

        # Set the initial position of the selection rectangle
        self.selection_rect.setGeometry(QtCore.QRect(self.start_pos, QtCore.QSize()))
        self.selection_rect.show()

    def update_selection(self, pos):
        self.current_pos = pos

        # Update the selection rectangle
        self.selection_rect.setGeometry(QtCore.QRect(self.start_pos, self.current_pos).normalized())

    def finish_selection(self, pos):
        self.selecting = False
        self.current_pos = pos

        # Hide the selection rectangle
        if self.selection_rect:
            self.selection_rect.hide()

        # Convert the selection rectangle to data coordinates
        x1, y1, x2, y2 = self.get_selection_data_coords()

        # Add the selection to the selection list
        if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
            self.add_selection_to_list(x1, y1, x2, y2)

    def get_selection_data_coords(self):
        if not self.start_pos or not self.current_pos:
            return None, None, None, None

        # Get the canvas size
        canvas = self.parent.overlay_plot.canvas()
        canvas_width = canvas.width()
        canvas_height = canvas.height()

        # Normalize the selection coordinates to [0, 1] range
        x1_norm = self.start_pos.x() / canvas_width
        y1_norm = self.start_pos.y() / canvas_height
        x2_norm = self.current_pos.x() / canvas_width
        y2_norm = self.current_pos.y() / canvas_height

        # Ensure the coordinates are in the correct order
        if x1_norm > x2_norm:
            x1_norm, x2_norm = x2_norm, x1_norm
        if y1_norm > y2_norm:
            y1_norm, y2_norm = y2_norm, y1_norm

        # Invert the y-coordinates because the canvas origin is at the top-left
        y1_norm = 1.0 - y1_norm
        y2_norm = 1.0 - y2_norm

        # Ensure the coordinates are within the [0, 1] range
        x1_norm = max(0.0, min(1.0, x1_norm))
        y1_norm = max(0.0, min(1.0, y1_norm))
        x2_norm = max(0.0, min(1.0, x2_norm))
        y2_norm = max(0.0, min(1.0, y2_norm))

        try:
            # Get the 2D histogram data and edges
            _, x_edges, y_edges = self.parent._histogram["2d"]

            # Convert normalized coordinates to bin indices
            x1_bin = int(x1_norm * (len(x_edges) - 1))
            y1_bin = int(y1_norm * (len(y_edges) - 1))
            x2_bin = int(x2_norm * (len(x_edges) - 1))
            y2_bin = int(y2_norm * (len(y_edges) - 1))

            # Convert bin indices to data values
            x1 = self.parent.bin_to_x_value(x1_bin, x_edges)
            y1 = self.parent.bin_to_y_value(y1_bin, y_edges)
            x2 = self.parent.bin_to_x_value(x2_bin, x_edges)
            y2 = self.parent.bin_to_y_value(y2_bin, y_edges)

            return x1, y1, x2, y2
        except (ValueError, KeyError, AttributeError):
            return None, None, None, None

    def add_selection_to_list(self, x1, y1, x2, y2):
        # Get the parameter indices for x and y axes
        x_idx = self.parent.plot_control.p1[0]
        y_idx = self.parent.plot_control.p2[0]

        # Get the parameter names for x and y axes
        x_name = self.parent.plot_control.p1[1]
        y_name = self.parent.plot_control.p2[1]

        # Ensure x values are ordered min to max
        if x1 > x2:
            x1, x2 = x2, x1

        # Ensure y values are ordered min to max
        if y1 > y2:
            y1, y2 = y2, y1

        # Add the x-axis selection
        self.parent.plot_control.addSelection(
            idx=x_idx,
            xmin=float(x1),
            xmax=float(x2),
            invert=False,
            enabled=True,
            name=f"{x_name}"
        )

        # Add the y-axis selection
        self.parent.plot_control.addSelection(
            idx=y_idx,
            xmin=float(y1),
            xmax=float(y2),
            invert=False,
            enabled=True,
            name=f"{y_name}"
        )
