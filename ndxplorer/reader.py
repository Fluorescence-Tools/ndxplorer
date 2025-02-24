from typing import List, Union
import json
import pandas as pd
from . data_source import DataSource
import pathlib

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
    Reads .bur files and any additional files specified.
    Now shows a separate PyQt progress dialog while processing.
    """
    # Make sure there's a QApplication running (required for any PyQt GUI)
    # If you already have a QApplication in your main script, remove this check.
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    base_path = pathlib.Path(base_path)

    if additional_endings is None:
        additional_endings = ["bg4", "br4", "by4"]

    path_bi4_bur = base_path / "bi4_bur"
    path_bur = base_path / "bur"

    # Attempt to find .bur files
    if path_bi4_bur.is_dir():
        bur_files = list(path_bi4_bur.glob("*.bur"))
        if not bur_files:
            print("No .bur files in 'bi4_bur'; falling back to 'bur' folder.")
            bur_files = list(path_bur.glob("*.bur"))
    else:
        print("'bi4_bur' folder does not exist; using 'bur' folder.")
        bur_files = list(path_bur.glob("*.bur"))

    if not bur_files:
        raise FileNotFoundError("No .bur files found in either 'bi4_bur' or 'bur'.")

    progress_window = ProgressWindow(
        title="File Processing",
        message="Processing Burst files...",
        max_value=len(bur_files),
    )
    progress_window.show()

    df_files = []

    # Process each .bur file
    for i, bur_file in enumerate(bur_files, start=1):
        # Read the main .bur file
        dfs = []
        df_bur = pd.read_csv(bur_file, sep="\t")
        if drop_last_column:
            df_bur.drop(df_bur.columns[-1], axis=1, inplace=True)
        dfs.append(df_bur)

        # Use the file stem to construct matching filenames
        fn_head = bur_file.stem

        # Read additional files
        for ending in additional_endings:
            extra_file = base_path / ending / f"{fn_head}.{ending}"
            if extra_file.exists():
                df_extra = pd.read_csv(extra_file, sep="\t")
                if drop_last_column:
                    df_extra.drop(df_extra.columns[-1], axis=1, inplace=True)
                dfs.append(df_extra)

        # Concatenate horizontally and apply row skipping
        combined_df = pd.concat(dfs, axis=1)
        df_files.append(combined_df[combined_df.index % skip_nth_row != 0])

        # -- Update the progress bar --
        progress_window.set_value(i)

        # Allow the GUI to refresh; avoid freezing
        QCoreApplication.processEvents()

    # Finished loop, set progress to max
    progress_window.set_value(len(bur_files))

    # Concatenate final DataFrame
    final_df = pd.concat(df_files, ignore_index=True)

    data_source = DataSource()
    data_source.data = final_df

    # Optionally close the progress window now that we're done
    progress_window.close()

    return data_source


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

