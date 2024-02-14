#!/usr/bin/python
try:
    import numpy as np
except ImportError:
    np = None
from setuptools import setup, find_packages

__name__ = "ndxplorer"
__author__ = "Thomas-Otavio Peulen"
__version__ = str(today.strftime("%y.%m.%d"))
__copyright__ = "Copyright (C) 2024 Thomas-Otavio Peulen"
__credits__ = ["Thomas-Otavio Peulen"]
__maintainer__ = "Thomas-Otavio Peulen"
__email__ = "thomas@peulen.xyz"
__url__ = "https://gitlab.peulen.xyz/tpeulen/ndxplorer/"
__license__ = 'GPL2.1'
__status__ = "Dev"
__description__ = "ndXplorer - An interactive tool to visualize multi dimensional data."
__app_id__ = "F25DCFFA-1234-4643-BC4F-2C3A20495931"
help_url = 'https://gitlab.peulen.xyz/tpeulen/ndxplorer/'
update_url = 'https://gitlab.peulen.xyz/tpeulen/ndxplorer/'


def dict_from_txt(fn):
    d = {}
    with open(fn) as f:
        for line in f:
            (key, val) = line.split()
            d[str(key)] = val
    return d


gui_scripts = dict_from_txt("./ndxplorer/entry_points/gui.txt")
console_scripts = dict_from_txt("./ndxplorer/entry_points/cmd.txt")


metadata = dict(
    name=__name__,
    version=__version__,
    license=__license__,
    description=__description__,
    author=__author__,
    author_email=__email__,
    app_id=__app_id__,
    url=__url__,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
    ],
    packages=find_packages(
        include=(__name__ + "*",)
    ),
    package_dir={
        __name__: __name__
    },
    include_package_data=True,
    package_data={
        '': [
            '*.json', '*.yaml',
            '*.ui',
            '*.png', '*.svg',
        ]
    },
    entry_points={
        "console_scripts": [
            "%s=%s" % (key, console_scripts[key]) for key in console_scripts
        ],
        "gui_scripts": [
            "%s=%s" % (key, gui_scripts[key]) for key in gui_scripts
        ]
    }
)

setup(**metadata)
