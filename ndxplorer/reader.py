from typing import List, Union
import pathlib

import json
import pandas as pd
from pandas.errors import EmptyDataError

from . data_source import DataSource

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QProgressBar, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QCoreApplication


class ProgressWindow(QDialog):
    def __init__(self, title="Progress", message="Processing...", max_value=100, parent=None):
        super().__init__(parent)

        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModal)  # Keep this window in front

        self.layout = QVBoxLayout()
        self.label = QLabel(message)
        self.progress_bar = QProgressBar()

        # Set the range of the progress bar (0 to max_value)
        self.progress_bar.setRange(0, max_value)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.progress_bar)
        self.setLayout(self.layout)

    def set_value(self, value: int):
        """
        Update the progress bar to the given value.
        """
        self.progress_bar.setValue(value)


def read_burst_analysis(
        base_path: Union[str, pathlib.Path] = "./test/mfd/burstwise_All 0.2500#30",
        skip_nth_row: int = 2,
        additional_endings: List[str] = None,
        drop_last_column: bool = True
) -> DataSource:
    """
    Reads .bur files and any additional files specified,
    including files that have only headers or are completely empty.
    Column names are preserved exactly as in the source files.
    """
    # ensure a QApplication
    app = QApplication.instance() or QApplication([])
    base_path = pathlib.Path(base_path)
    additional_endings = additional_endings or ["bg4", "br4", "by4", "bv4"]

    # locate .bur files
    dir_main = base_path / "bi4_bur"
    dir_fallback = base_path / "bur"
    if dir_main.is_dir():
        bur_files = list(dir_main.glob("*.bur")) or list(dir_fallback.glob("*.bur"))
    else:
        bur_files = list(dir_fallback.glob("*.bur"))
    if not bur_files:
        raise FileNotFoundError("No .bur files found in either 'bi4_bur' or 'bur'.")

    # helper: read CSV or build header‐only DataFrame
    def _read_file(path: pathlib.Path) -> pd.DataFrame:
        try:
            df = pd.read_csv(path, sep="\t")
        except EmptyDataError:
            # try to pull headers even if there's no data rows
            with open(path, 'r') as f:
                first = f.readline().strip()
            cols = first.split("\t") if first else []
            df = pd.DataFrame(columns=cols)
        return df

    # progress dialog
    progress = ProgressWindow(
        title="File Processing",
        message="Processing Burst files...",
        max_value=len(bur_files),
    )
    progress.show()

    pieces = []
    for idx, bur_file in enumerate(bur_files, start=1):
        # --- main .bur ---
        df_main = _read_file(bur_file)
        if drop_last_column and df_main.shape[1] > 1:
            df_main = df_main.iloc[:, :-1]
        dfs = [df_main]

        # --- extras ---
        stem = bur_file.stem
        for ending in additional_endings:
            extra = base_path / ending / f"{stem}.{ending}"
            if not extra.exists():
                continue
            df_extra = _read_file(extra)
            # if truly empty (no cols), skip
            if df_extra.shape[1] == 0:
                continue
            if drop_last_column and df_extra.shape[1] > 1:
                df_extra = df_extra.iloc[:, :-1]
            dfs.append(df_extra)

        # horizontal concat (index‐aligned)
        combined = pd.concat(dfs, axis=1)
        # —————————————————————————————————————————————
        # drop any duplicate columns now (keep the first occurrence)
        combined = combined.loc[:, ~combined.columns.duplicated()]
        # —————————————————————————————————————————————

        # skip every Nth row
        if skip_nth_row > 1:
            combined = combined[ combined.index % skip_nth_row != 0 ]

        pieces.append(combined)

        progress.set_value(idx)
        QCoreApplication.processEvents()

    progress.set_value(len(bur_files))

    # final vertical concat
    final_df = (
        pd.concat(pieces, ignore_index=True)
        if any(len(df) for df in pieces)
        else pieces[0].iloc[0:0]  # zero‐row with correct cols if no data at all
    )

    ds = DataSource()
    ds.data = final_df

    progress.close()
    return ds

def read_csv_sampling(filenames, sep='\t'):
    # type: (List[str])->(DataSource)
    with open(filenames[0], "r") as fp:
        l = fp.readline()
        pn = l.split("\t")
    values = list()
    for filename in filenames:
        df = pd.read_csv(filename, sep=sep)
        values.append(df)
    data = pd.concat(values)
    return DataSource(
        data=data,
        parameter_names=pn
    )


def read_csv(filenames):
    df_files = list()
    for filename in filenames:
        df = pd.read_csv(filename, sep="\t")
        df_files.append(df)
    dfs = pd.concat(df_files)
    dfn = dfs.select_dtypes(['number'])
    ds = DataSource()
    ds.data = dfn
    return ds

