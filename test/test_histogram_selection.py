import numpy as np
from ndxplorer.core.data_source import DataSource, RectangularDataSelection

np.random.seed(0)

data = np.vstack(
    [
        np.random.normal(4, 0.1, 5),
        np.random.normal(4, 0.1, 5)
    ]
)

d = DataSource(["x", "y"], data)
s = RectangularDataSelection(0, 3.95, 4.05, False, True)
m = d.get_mask([s])

assert m.shape == (2, 5)
assert m.dtype == bool

