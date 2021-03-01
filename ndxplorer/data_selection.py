from __future__ import print_function
import sys
import abc
import numpy as np


class DataSelection(object):

    @abc.abstractmethod
    def get_mask(
            self,
            data  # type: np.ndarray
    ):
        # type: (np.ndarray) -> np.ndarray
        pass


class RectangularDataSelection(DataSelection):

    def __init__(
            self,
            parameter_idx,   # type: int
            lower,  # type: float
            upper,   # type: float
            invert=False,   # type: bool
            enabled=True,   # type: bool
            name=None  # type: str
    ):
        self.parameter_idx = parameter_idx
        self.lower = lower
        self.upper = upper
        self.invert = invert
        self.enabled = enabled
        self.name = name

    def __str__(self):
        s = "RectangularDataSelection:\n"
        s += "Bounds: %s, %s\n" % (self.lower, self.upper)
        s += "Invert: %s\n" % self.invert
        s += "Enabled: %s\n" % self.enabled
        return s

    def get_mask(
            self,
            data  # type: np.ndarray
    ):
        # type: (np.ndarray) -> np.ndarray
        n_parameter, n_data_points = data.shape
        mask = np.ma.make_mask_none((n_parameter, n_data_points))
        s = self
        if s.parameter_idx < n_parameter:
            if s.enabled:
                if s.invert:
                    mask[:, :] |= np.logical_and(data[s.parameter_idx, :] > s.lower, data[s.parameter_idx, :] < s.upper)
                else:
                    mask[:, :] |= np.logical_or(data[s.parameter_idx, :] < s.lower, data[s.parameter_idx, :] > s.upper)
        else:
            print("Parameter with idx %s exceeds dimension of dataset." % s.parameter_idx, file=sys.stderr)
        return mask

