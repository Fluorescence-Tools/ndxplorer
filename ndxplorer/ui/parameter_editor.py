"""
This example demonstrates the use of pyqtgraph's parametertree system. This provides
a simple way to generate user interfaces that control sets of parameters. The example
demonstrates a variety of different parameter types (int, float, list, etc.)
as well as some customized parameter types

"""
from typing import Dict, List
from qtpy import QtGui, QtWidgets
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.parametertree.parameterTypes import ListParameter
import re
import json
import sys
from collections import OrderedDict


def dict2pt(target, origin):
    """Creates an array from a dictionary that can be used to initialize a pyqtgraph parameter-tree

    :param target:
    :param origin:
    :return:
    """
    for i, key in enumerate(origin.keys()):
        if key.endswith('_options'):
            target[-1]['values'] = origin[key]
            target[-1]['type'] = 'list'
            continue
        d = dict()
        d['name'] = key
        if isinstance(origin[key], dict):
            d['type'] = 'group'
            d['children'] = dict2pt(list(), origin[key])
        else:
            value = origin[key]
            type = value.__class__.__name__
            if type == 'unicode':
                type = 'str'
            iscolor = re.search(r'^#(?:[0-9a-fA-F]{3}){1,2}$', str(value))
            if iscolor:
                type = 'color'
            if type == 'float':
                # Use adaptive stepping for floats; pyqtgraph SpinBox supports 'dec' and 'minStep'
                d['dec'] = True  # enable decade-based adaptive step size
                import math
                av = abs(float(value))
                # Choose a reasonable minimum step relative to current magnitude
                min_step = 10 ** (math.floor(math.log10(av)) - 4)
                d['minStep'] = min_step
            elif type == 'int':
                # Ensure integer stepping for integers
                d['int'] = True
                d['step'] = 1
            d['type'] = type
            d['value'] = value
            d['expanded'] = False
        target.append(d)
    return target


def pt2dict(parameter_tree, target=OrderedDict()):
    """Converts a pyqtgraph parameter tree to an ordinary dictionary that could be saved
    as JSON file

    :param parameter_tree:
    :param target:
    :return:
    """
    children = parameter_tree.children()
    for i, child in enumerate(children):
        if child.type() == "action":
            continue
        if not child.children():
            value = child.opts['value']
            name = child.name()
            if isinstance(value, QtGui.QColor):
                value = str(value.name())
            if isinstance(child, ListParameter):
                target[name + '_options'] = child.opts['values']
            target[name] = value
        else:
            target[child.name()] = pt2dict(child, OrderedDict())
    return target


class ParameterEditor(QtWidgets.QWidget):

    def __init__(
            self,
            json_file=None,  # type: str
            parent=None,  # type: QtWidgets.QWidget
            callback=None  # type: callable
    ):
        super(ParameterEditor, self).__init__(parent)

        self._dict = dict()  # type: Dict
        self._p = None
        self._json_file = None  # type: str
        self._target = list()  # type: List

        self._json_file = json_file
        self.json_file = json_file

        self._p = Parameter.create(
            name='params',
            type='group',
            children=dict2pt(self._target, self._dict),
            expanded=True
        )

        self.setWindowTitle("Configuration: %s" % self.json_file)
        t = ParameterTree()
        t.setParameters(self._p, showTop=False)
        layout = QtWidgets.QGridLayout()
        layout.addWidget(t, 1, 0, 1, 1)
        self.setLayout(layout)
        self.setSizePolicy(
            QtWidgets.QSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Expanding
            )
        )

        self.set_callback(callback)

    def set_callback(self, cb, root=None):
        #  type: (callable, Parameter)->()
        if root is None:
            root = self._p
        root.sigValueChanged.connect(cb)
        for ch in root.children():
            self.set_callback(cb, ch)

    @property
    def dict(self):
        if self._p is not None:
            return pt2dict(self._p, OrderedDict())
        else:
            return self._dict

    @property
    def parameter_dict(self):
        od = OrderedDict(self.dict)
        params = dict2pt(list(), od)
        params.append(
            {
                'name': 'Save',
                'type': 'action'
            }
        )
        return params

    @property
    def json_file(self):
        return self._json_file

    @json_file.setter
    def json_file(self, v):
        import pathlib
        import logging
        if v is not None and pathlib.Path(v).exists():
            with open(v, 'r') as fp:
                self._dict = json.load(fp, object_pairs_hook=OrderedDict)
        else:
            logging.warning(f"Parameter file not found: {v}")
            # Fallback to packaged defaults if possible
            try:
                default_v = pathlib.Path(__file__).parent.parent / "settings" / "mfd.constants.json"
                if default_v.exists():
                    with open(default_v, 'r') as fp:
                        self._dict = json.load(fp, object_pairs_hook=OrderedDict)
            except Exception as e:
                logging.error(f"Failed to load default parameters: {e}")
        self._json_file = v


def main():
    app = QtWidgets.QApplication(sys.argv)
    target = dict()
    pt = ParameterEditor(
        target=target,
        json_file="settings/mfd.constants.json"
    )
    pt.show()
    app.exec_()


if __name__ == "__main__":
    main()

