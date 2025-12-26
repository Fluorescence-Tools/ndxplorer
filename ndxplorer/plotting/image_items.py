"""
Custom image item classes for ndxplorer visualization.
"""

import numpy as np
from qtpy.QtGui import QImage

try:
    import guiqwt.image
    GUIQWT_AVAILABLE = True
    BaseImageItem = guiqwt.image.ImageItem
except ImportError:
    GUIQWT_AVAILABLE = False
    # Create a dummy base class for when guiqwt is not available
    class BaseImageItem:
        def __init__(self, *args, **kwargs):
            self.data = None
            self._offscreen = None
            self._image = None
            
        def set_data(self, data):
            self.data = data
            
        def draw_image(self, *args, **kwargs):
            pass
            
        def draw_border(self, *args, **kwargs):
            pass
            
        def boundingRect(self):
            # Return a dummy rectangle
            class DummyRect:
                def getCoords(self):
                    return 0, 0, 100, 100
            return DummyRect()
            
        def notify_new_offscreen(self):
            pass

# Custom ImageItem class that fixes the float to int conversion issue
class FixedImageItem(BaseImageItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._matplotlib_colormap = None
        self._matplotlib_norm = None
        
    def set_matplotlib_colormap(self, colormap_name, vmin=None, vmax=None):
        """Set matplotlib colormap for fallback when guiqwt is not available."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import Normalize
            
            self._matplotlib_colormap = plt.get_cmap(colormap_name)
            if vmin is not None and vmax is not None:
                self._matplotlib_norm = Normalize(vmin=vmin, vmax=vmax)
            else:
                self._matplotlib_norm = None
            return True
        except ImportError:
            return False
    
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
        if (not hasattr(self, '_offscreen')) or getattr(self, '_offscreen', None) is None or getattr(self, '_offscreen', None).shape != (H, W):
            self._offscreen = np.empty((H, W), np.uint32)
            self._image = QImage(self._offscreen, W, H, QImage.Format_ARGB32)
            # Attach numpy array for compatibility with guiqwt expectations
            self._image.ndarray = self._offscreen
            self.notify_new_offscreen()

        # Apply matplotlib colormap if available and guiqwt is not
        if (self._matplotlib_colormap is not None and 
            hasattr(self, 'data') and self.data is not None and
            not GUIQWT_AVAILABLE):
            try:
                # Apply colormap to data
                data = self.data
                if self._matplotlib_norm is not None:
                    colored_data = self._matplotlib_colormap(self._matplotlib_norm(data))
                else:
                    colored_data = self._matplotlib_colormap(data)
                
                # Convert to ARGB format for display
                rgba_uint8 = (colored_data * 255).astype(np.uint8)
                argb_uint32 = (
                    (255 << 24) |  # Alpha channel (fully opaque)
                    (rgba_uint8[:, :, 0] << 16) |  # Red channel
                    (rgba_uint8[:, :, 1] << 8) |   # Green channel
                    (rgba_uint8[:, :, 2])          # Blue channel
                )
                
                # Update the offscreen buffer with colored data
                h, w = argb_uint32.shape
                if self._offscreen.shape != (h, w):
                    self._offscreen = np.empty((h, w), np.uint32)
                    self._image = QImage(self._offscreen, w, h, QImage.Format_ARGB32)
                    self._image.ndarray = self._offscreen
                    self.notify_new_offscreen()
                self._offscreen[:] = argb_uint32
            except Exception as e:
                # Fallback to default rendering if colormap application fails
                pass

        self.draw_image(painter, canvasRect, (i1, j1, i2, j2), dest, xMap, yMap)
        self.draw_border(painter, xMap, yMap, canvasRect)