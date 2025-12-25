"""Utility helpers to keep the file-loading logic for NDXplorer separate from plot_main."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

from qtpy import QtWidgets

from ..logging_config import logging
from ..io import reader

if False:  # pragma: no cover - circular import safety for type checkers
    from .plot_main import NDXplorer


def _ensure_sequence(handles: Optional[Sequence[str]]) -> Sequence[str]:
    """Normalize Qt's return types (tuple/list) to a plain tuple."""
    if handles is None:
        return ()
    if isinstance(handles, (list, tuple)):
        return tuple(handles)
    return (handles,)


def _update_working_path(ndxplorer: "NDXplorer", first_selection: Optional[str]) -> None:
    """Keep ndxplorer.working_path synced with the most recent selection."""
    if not first_selection:
        return
    try:
        dir_path = str(Path(first_selection).parent)
    except Exception as exc:  # pragma: no cover - defensive
        logging.debug("Could not derive working path: %s", exc)
        return
    ndxplorer.working_path = dir_path


def _handle_append(ndxplorer: "NDXplorer", new_source, merge_mode: str) -> None:
    """Append/replace loaded data followed by UI updates."""
    if (
        hasattr(ndxplorer, "_data_source")
        and ndxplorer._data_source is not None
        and not ndxplorer._data_source.empty
    ):
        if ndxplorer._data_source.merge(new_source, mode=merge_mode):
            ndxplorer.update()
    else:
        ndxplorer._data_source = new_source
        ndxplorer.update()


def open_files(
    ndxplorer: "NDXplorer",
    file_handles: Optional[Sequence[str]] = None,
    file_type: Optional[str] = None,
    append: bool = False,
    merge_mode: str = "columns",
) -> None:
    """Central entry point for all data-loading actions."""
    logging.info("NDXplorer: Opening files..")
    logging.debug("File handles: %s", file_handles)
    logging.debug("File type: %s", file_type)
    logging.debug("Append mode: %s", append)
    logging.debug("Merge mode: %s", merge_mode)

    file_handles_seq = _ensure_sequence(file_handles)
    reader_input = file_handles_seq
    working_path = str(ndxplorer.working_path)

    # --- Sampling / ER4 ----------------------------------------------------
    if file_type in {"cs_sampling", "er4"}:
        if not file_handles_seq:
            file_handles_seq, _ = QtWidgets.QFileDialog.getOpenFileNames(
                ndxplorer,
                "ChiSurf sampling files",
                working_path,
                "Sampling files (*.*)",
            )
        _update_working_path(ndxplorer, file_handles_seq[0] if file_handles_seq else None)
        logging.info("Opening files (%s): %s", file_type, file_handles_seq)
        data_reader = reader.read_csv_sampling

    # --- Burst directories -------------------------------------------------
    elif file_type == "burst_dir":
        if not file_handles_seq:
            directory = QtWidgets.QFileDialog.getExistingDirectory(
                ndxplorer, "Open burst analysis folder", working_path
            )
            file_handles_seq = (directory,) if directory else ()
        if file_handles_seq:
            selected_dir = file_handles_seq[0]
            ndxplorer.working_path = str(selected_dir)
            reader_input = selected_dir
        else:
            reader_input = ()
        data_reader = reader.read_burst_analysis

    # --- HDF5 / zipped HDF5 ------------------------------------------------
    elif file_type == "mfd_hdf5":
        if not file_handles_seq:
            file_handles_seq, _ = QtWidgets.QFileDialog.getOpenFileNames(
                ndxplorer,
                "MFD HDF5 files",
                working_path,
                "HDF5 files (*.h5 *.hdf5);;ZIP files (*.zip);;All Files (*.*)",
            )

        _update_working_path(ndxplorer, file_handles_seq[0] if file_handles_seq else None)
        logging.info("Opening MFD HDF5/Zip files: %s", file_handles_seq)

        if not file_handles_seq:
            return

        combined_data_source = None
        for file_path in file_handles_seq:
            path_str = str(file_path)
            is_zip = path_str.lower().endswith(".zip")

            if is_zip:
                try:
                    import zipfile as _zip

                    with _zip.ZipFile(path_str, "r") as zf:
                        names = zf.namelist()
                    has_h5 = any(
                        name.lower().endswith((".h5", ".hdf5")) for name in names
                    )
                    logging.debug("ZIP '%s' contains HDF5: %s", path_str, has_h5)
                except Exception as exc:
                    logging.debug("Could not inspect zip '%s': %s", path_str, exc)
                    has_h5 = False

                temp_ds = (
                    reader.read_mfd_hdf5([path_str])
                    if has_h5
                    else reader.read_burst_analysis(path_str)
                )
            else:
                temp_ds = reader.read_mfd_hdf5([path_str])

            if combined_data_source is None:
                combined_data_source = temp_ds
            else:
                combined_data_source.merge(temp_ds, mode=merge_mode)

        if append:
            _handle_append(ndxplorer, combined_data_source, merge_mode)
        else:
            ndxplorer._data_source = combined_data_source
            ndxplorer.update()

        _apply_axes_and_refresh(ndxplorer)
        logging.debug("Handled mfd_hdf5; returning before generic loader.")
        return

    # --- Generic CSV loader ------------------------------------------------
    else:
        if not file_handles_seq:
            file_handles_seq, _ = QtWidgets.QFileDialog.getOpenFileNames(
                ndxplorer,
                "Comma separated value files",
                working_path,
                "Text files (*.*)",
            )
        _update_working_path(ndxplorer, file_handles_seq[0] if file_handles_seq else None)
        data_reader = reader.read_csv

    if reader_input:
        logging.info("Opening files (%s): %s", file_type or "csv", file_handles_seq)
        new_data_source = data_reader(reader_input)
        if append:
            _handle_append(ndxplorer, new_data_source, merge_mode)
        else:
            ndxplorer._data_source = new_data_source
            ndxplorer.update()

    _apply_axes_and_refresh(ndxplorer)


def _apply_axes_and_refresh(ndxplorer: "NDXplorer") -> None:
    """Shared tail for open operations: apply axes + refresh plots."""
    img_applied = ndxplorer.check_and_set_image_axes()
    if not img_applied:
        try:
            ndxplorer.apply_default_axes_from_settings()
        except Exception as exc:  # pragma: no cover - defensive
            logging.debug("Could not apply default axes: %s", exc)
    ndxplorer.on_auto_contrast()


def show_merge_dialog(
    ndxplorer: "NDXplorer", title: str
) -> Optional[tuple[bool, str]]:
    """Modal dialog prompting the user for merge behavior."""
    logging.debug("show_merge_dialog")
    dialog = QtWidgets.QDialog(ndxplorer)
    dialog.setWindowTitle(title)
    layout = QtWidgets.QVBoxLayout()

    label = QtWidgets.QLabel("How do you want to merge the new data?")
    layout.addWidget(label)

    replace_rb = QtWidgets.QRadioButton("Replace existing data")
    append_columns_rb = QtWidgets.QRadioButton(
        "Append as columns (add new columns, rows must match)"
    )
    append_rows_rb = QtWidgets.QRadioButton(
        "Append as rows (add new rows of existing columns)"
    )
    replace_rb.setChecked(True)

    layout.addWidget(replace_rb)
    layout.addWidget(append_columns_rb)
    layout.addWidget(append_rows_rb)

    buttons = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
    )
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)

    dialog.setLayout(layout)
    result = dialog.exec_()
    if result != QtWidgets.QDialog.Accepted:
        return None

    append = append_columns_rb.isChecked() or append_rows_rb.isChecked()
    merge_mode = "columns"
    if append_columns_rb.isChecked():
        merge_mode = "columns"
    elif append_rows_rb.isChecked():
        merge_mode = "rows"
    return append, merge_mode
