import numpy as np
import chisurf.tools.ndexploder.data_selection
import chisurf.tools.ndexploder.data_source

np.random.seed(0)

data = np.vstack(
            [
                np.random.normal(4, 0.1, 5),
                np.random.normal(4, 0.1, 5)
            ]
        )

d = chisurf.tools.plot_histograms.data_source.DataSource(["x", "y"], data)
s = chisurf.tools.plot_histograms.data_selection.RectangularDataSelection(0, 3.95, 4.05, False, True)
m = d.get_mask([s])

mask_ref = np.array(
    [[True, False, True, True, True],
     [True, False, True, True, True]], dtype=bool
)
np.allclose(m, mask_ref) == True

