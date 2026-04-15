"""Utilities related to the working path UI interactions."""

from __future__ import annotations

from pathlib import Path

from qtpy import QtWidgets

from ..logging_config import logging


def install_working_path_drop(ndxplorer) -> None:
    """Enable drag/drop on the main window for dirs/HDF5/CSV."""
    logging.debug("_install_working_path_drop")
    try:
        ndxplorer.setAcceptDrops(True)
    except Exception:
        pass

    def dragEnterEvent(event):
        try:
            mime = event.mimeData()
            if mime and mime.hasUrls():
                urls = mime.urls()
                if urls:
                    path = Path(urls[0].toLocalFile())
                    if path.exists() and (path.is_dir() or path.suffix.lower() in (".h5", ".hdf5", ".csv", ".er4")):
                        event.acceptProposedAction()
                        return
            event.ignore()
        except Exception as exc:
            logging.debug("dragEnterEvent error: %s", exc)
            event.ignore()

    def dragMoveEvent(event):
        """Need to override dragMoveEvent to accept drops on the main window."""
        event.acceptProposedAction()

    def dropEvent(event):
        try:
            mime = event.mimeData()
            if not (mime and mime.hasUrls()):
                event.ignore()
                return
            paths = [Path(u.toLocalFile()) for u in mime.urls()]
            dirs = [p for p in paths if p.exists() and p.is_dir()]
            if dirs:
                folder = dirs[0]
                event.acceptProposedAction()
                try:
                    ndxplorer.lineEditWorkingPath.setText(str(folder))
                except Exception:
                    pass
                try:
                    ndxplorer.open_files(file_type="burst_dir", file_handles=str(folder), append=False)
                except Exception as exc:
                    logging.error("Failed to open burst analysis folder from drop: %s", exc)
                return

            files = [p for p in paths if p.exists() and p.is_file()]
            csvs = [str(p) for p in files if p.suffix.lower() == ".csv"]
            er4s = [str(p) for p in files if p.suffix.lower() == ".er4"]
            h5s = [str(p) for p in files if p.suffix.lower() in (".h5", ".hdf5")]

            if er4s:
                event.acceptProposedAction()
                try:
                    ndxplorer.onOpenChiSurfSampling(filenames=er4s, append=False, merge_mode="columns")
                except Exception as exc:
                    logging.error("Failed to open .er4 from drop: %s", exc)
                return

            if csvs:
                event.acceptProposedAction()
                try:
                    ndxplorer.onOpenCsv(None, filenames=csvs, append=False, merge_mode="columns")
                except Exception as exc:
                    logging.error("Failed to open CSV from drop: %s", exc)
                return

            if h5s:
                event.acceptProposedAction()
                try:
                    ndxplorer.onOpenMfdHdf5(None, filenames=h5s, append=False, merge_mode="columns")
                except Exception as exc:
                    logging.error("Failed to open HDF5 from drop: %s", exc)
                return

            event.ignore()
        except Exception as exc:
            logging.debug("dropEvent error: %s", exc)
            event.ignore()

    ndxplorer.dragEnterEvent = dragEnterEvent
    ndxplorer.dragMoveEvent = dragMoveEvent
    ndxplorer.dropEvent = dropEvent
