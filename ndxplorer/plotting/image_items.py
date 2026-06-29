"""
FixedImageItem — compatibility shim retained for public API stability.

The real 2D display backend is PGImageWidget (pyqtgraph). This class exists
only so that ``from ndxplorer import FixedImageItem`` does not break external
code that imported it during the guiqwt→pyqtgraph migration.
"""

import numpy as np
from qtpy.QtGui import QImage


class FixedImageItem:
    """Compatibility shim. Does nothing — the real display is PGImageWidget."""

    def __init__(self, *args, **kwargs):
        self.data = None

    def set_data(self, data):
        self.data = data