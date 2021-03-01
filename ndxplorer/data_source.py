from __future__ import print_function
from typing import List, Dict

import sys
import yaml
import numpy as np
import pandas as pd

from collections import OrderedDict
from . data_selection import DataSelection


def compute_values(
        d,  # type: pd.DataFrame
        constants,  # type: Dict[str, float]
        equations,  # type: List[Dict[str, str]]
        equation_json_fn=None,  # type: str
        engine='numexpr'
):
    # type: (pd.DataFrame, Dict[str, float], List[Dict[str, str]], str, str) -> None
    c = constants
    if equation_json_fn is not None:
        with open(equation_json_fn, "r") as fp:
            equations = yaml.loads(fp.read(), object_pairs_hook=OrderedDict)
    print(equations)
    for eq in equations:
        for key in eq:
            print("Compute value: %s" % key)
            try:
                d[key] = pd.eval(eq[key], engine=engine)
            except:
                print("Could not compute: %s" % key, file=sys.stderr)


class DataSource(object):

    _parameter_names = list()  # type: List[str]
    _data = None  # type: pd.DataFrame
    _data_numeric = None  # type: pd.DataFrame

    def __init__(
            self,
            parameter_names=None,  # type: List[str]
            data=None  # type: np.ndarray
    ):
        # Data
        if isinstance(data, np.ndarray):
            self.data = pd.DataFrame(data, columns=parameter_names)
        elif isinstance(data, pd.DataFrame):
            self.data = data
        else:
            self.data = pd.DataFrame()
        # Parameter names
        if isinstance(parameter_names, list):
            self._parameter_names = parameter_names
        else:
            self._parameter_names = list(self._data.columns)

    def __str__(self):
        return self._data.__str__()

    @property
    def parameter_names(self):
        # type: () -> List[str]
        return self._parameter_names

    @property
    def values(self):
        # type: () -> np.ndarray
        v = np.array(self._data_numeric).T
        return v

    def clear(self):
        self.data = pd.DataFrame()

    def compute_columns(
            self,
            constants,  # type: Dict[str, float]
            equations,  # type: List[Dict[str, str]]
            equation_json_fn=None  # type: str
    ):
        compute_values(
            d=self.data,
            constants=constants,
            equations=equations,
            equation_json_fn=equation_json_fn
        )
        # use property to update dependent attributes
        self.data = self.data

    @property
    def empty(self):
        # type: ()->(bool)
        return self._data.empty

    def get_mask(
            self,
            selections,  # type: List[DataSelection]
            idxs=None,  # type: List[int]
            mask_nan=True,  # type: bool
            mask_inf=True  # type: bool
    ):
        # type: (List[DataSelection]) -> np.ndarray
        """

        :param selections: A list of selections
        :return: mask - masked values are True
        """
        if idxs is None:
            idxs = list()
        d = self.values
        mask = np.ma.make_mask_none(d.shape)
        for s in selections:
            mask |= s.get_mask(d)
        for idx in idxs:
            if idx < 0:
                continue
            if mask_nan:
                mask[:, :] |= np.isnan(d[idx, :])
            if mask_inf:
                mask[:, :] |= np.isinf(d[idx, :])
        return mask

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, v):
        self._data = v
        self._parameter_names = list(self._data.columns)
        self._data_numeric = v.apply(pd.to_numeric, errors='coerce')

    @property
    def size(self):
        return self.values.shape[1]
