"""
Scale control mixin for log/linear axis scales.

Extracts scale-related functionality from SurfacePlotWidget.
"""


class ScaleControlMixin:
    """
    Mixin providing scale control properties and methods.
    
    Manages:
    - X/Y/Z axis scale (log vs linear)
    - Scale checkbox state
    - Scale property accessors
    """
    
    def _get_scale(self, checkbox):
        """Helper to get scale value from checkbox."""
        return "log" if checkbox.isChecked() else "lin"

    def _set_scale(self, checkbox, value):
        """Helper to set scale value on checkbox."""
        checkbox.setChecked(value == "log")

    @property
    def scale_x(self):
        """Get X axis scale ('log' or 'lin')."""
        return self._get_scale(self.checkBoxLogX)

    @scale_x.setter
    def scale_x(self, v):
        """Set X axis scale."""
        self._set_scale(self.checkBoxLogX, v)

    @property
    def scale_y(self):
        """Get Y axis scale ('log' or 'lin')."""
        return self._get_scale(self.checkBoxLogY)

    @scale_y.setter
    def scale_y(self, v):
        """Set Y axis scale."""
        self._set_scale(self.checkBoxLogY, v)

    @property
    def scale_z(self):
        """Get Z axis scale ('log' or 'lin')."""
        return self._get_scale(self.checkBoxLogZ)

    @scale_z.setter
    def scale_z(self, v):
        """Set Z axis scale."""
        self._set_scale(self.checkBoxLogZ, v)
