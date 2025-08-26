"""
Custom image item classes for ndxplorer visualization.
"""

import numpy as np
import guiqwt.image
from qtpy.QtGui import QImage

# Custom ImageItem class that fixes the float to int conversion issue
class FixedImageItem(guiqwt.image.ImageItem):
    def draw(self, painter, xMap, yMap, canvasRect):
        x1, y1, x2, y2 = canvasRect.getCoords()
        i1, i2 = xMap.invTransform(x1), xMap.invTransform(x2)
        j1, j2 = yMap.invTransform(y1), yMap.invTransform(y2)

        xl, yt, xr, yb = self.boundingRect().getCoords()
        dest = (
            xMap.transform(xl),
            yMap.transform(yt),
            xMap.transform(xr) + 1,
            yMap.transform(yb) + 1,
        )

        # Use canvas width and height for W and H (ensure at least 1 to avoid zero-size buffers on resize)
        # Guard against transient negative/zero sizes during interactive resizing
        try:
            cw = float(canvasRect.width())
            ch = float(canvasRect.height())
        except Exception:
            cw, ch = 1.0, 1.0
        W = max(1, int(round(max(0.0, cw))))
        H = max(1, int(round(max(0.0, ch))))

        # Initialize or resize the offscreen buffer safely
        if (not hasattr(self, '_offscreen')) or getattr(self, '_offscreen', None) is None or getattr(self._offscreen, 'shape', None) != (H, W):
            self._offscreen = np.empty((H, W), np.uint32)
            self._image = QImage(self._offscreen, W, H, QImage.Format_ARGB32)
            # Attach numpy array for compatibility with guiqwt expectations
            self._image.ndarray = self._offscreen
            self.notify_new_offscreen()

        self.draw_image(painter, canvasRect, (i1, j1, i2, j2), dest, xMap, yMap)
        self.draw_border(painter, xMap, yMap, canvasRect)