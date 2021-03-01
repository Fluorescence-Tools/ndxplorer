from __future__ import print_function
from typing import List
import os
import numpy as np
from data_selection import DataSelection
from data_source import DataSource


def save_burst_ids(
        folder_name,  # type: str
        selections,  # type: List[DataSelection]
        data_source  # type: DataSource
):
    print("save_burst_ids")
    df = data_source.data
    mask = data_source.get_mask(
        selections=selections
    )
    mask_flat = np.sum(mask, axis=0).astype(np.bool)
    mas = np.broadcast_to(mask_flat, (df.shape[1], df.shape[0]))
    dm = df.mask(mas.T)
    dm = dm.loc[dm['First File'] == dm['Last File']]
    grouped = dm.groupby('First File')
    for i, g in grouped:
        fn, ext = os.path.splitext(i)
        fn = os.path.join(folder_name, fn + "_0" + ext + ".bst")
        a = np.vstack([g["First Photon"], g["Last Photon"]]).astype(np.int)
        np.savetxt(fn, a.T, fmt='%i', delimiter='\t')
    pass


