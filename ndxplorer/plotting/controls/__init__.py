"""
Control components for NDXplorer plotting.

Decomposes the monolithic SurfacePlotWidget into focused, reusable components.
"""

from .scale_control import ScaleControlMixin
from .axis_control import AxisControlMixin
from .histogram_control import HistogramControlMixin

__all__ = [
    'ScaleControlMixin',
    'AxisControlMixin', 
    'HistogramControlMixin',
]
