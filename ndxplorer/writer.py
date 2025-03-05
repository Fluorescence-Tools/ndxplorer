from __future__ import print_function
from typing import List
from pathlib import Path
try:
    from chisurf import logging
except ImportError:
    import logging
import numpy as np
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QProgressBar, QLabel, QApplication
from PyQt5.QtCore import Qt, QCoreApplication
from .data_source import DataSource, DataSelection


class ProgressWindow(QDialog):
    def __init__(self, title="Progress", message="Processing...", max_value=100, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModal)
        self.layout = QVBoxLayout()
        self.label = QLabel(message)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, max_value)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.progress_bar)
        self.setLayout(self.layout)

    def set_value(self, value: int):
        self.progress_bar.setValue(value)


def save_burst_ids(
        folder_name: str,
        selections: List[DataSelection],
        data_source: DataSource
):
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    folder_path = Path(folder_name)
    folder_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving burst IDs to {folder_path}")

    df = data_source.data
    mask = data_source.get_mask(selections=selections)

    mask_flat = np.sum(mask, axis=0).astype(bool)
    mas = np.broadcast_to(mask_flat, (df.shape[1], df.shape[0]))
    dm = df.mask(mas.T)

    dm = dm.loc[dm['First File'] == dm['Last File']]
    grouped = dm.groupby('First File')

    total_files = len(grouped)
    progress_window = ProgressWindow(title="Saving Files", message="Saving Burst ID files...", max_value=total_files)
    progress_window.show()

    for i, (filename, g) in enumerate(grouped, start=1):
        fn = Path(filename).name
        ext = Path(filename).suffix
        bst_file = folder_path / f"{fn}.bst"

        a = np.vstack([g["First Photon"], g["Last Photon"]]).astype(int)
        np.savetxt(bst_file, a.T, fmt='%i', delimiter='\t')

        logging.info(f"Saved burst ID file: {bst_file}")
        progress_window.set_value(i)
        QCoreApplication.processEvents()

    progress_window.set_value(total_files)
    progress_window.close()