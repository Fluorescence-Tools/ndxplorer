"""Helpers for opening datasets and handling merge dialogs."""

from __future__ import annotations

from typing import List, Optional

from ..logging_config import logging
from ..io import file_operations


def show_merge_dialog(ndxplorer, title: str):
    """Delegate to file_operations.show_merge_dialog with ndxplorer context."""
    return file_operations.show_merge_dialog(ndxplorer, title)


def open_files(
    ndxplorer,
    file_handles: Optional[List[str]] = None,
    file_type: Optional[str] = None,
    append: bool = False,
    merge_mode: str = "columns",
):
    """Proxy to file_operations.open_files."""
    return file_operations.open_files(
        ndxplorer,
        file_handles=file_handles,
        file_type=file_type,
        append=append,
        merge_mode=merge_mode,
    )


def open_csv(ndxplorer, filenames: Optional[List[str]] = None, append: bool = False, merge_mode: str = "columns"):
    logging.debug("open_csv")
    if (
        filenames is None
        and getattr(ndxplorer, "_data_source", None) is not None
        and not ndxplorer._data_source.empty
    ):
        result = show_merge_dialog(ndxplorer, "Open CSV Files")
        if result is None:
            return
        append, merge_mode = result
    open_files(ndxplorer, file_handles=filenames, file_type="csv", append=append, merge_mode=merge_mode)


def open_chisurf_sampling(
    ndxplorer,
    filenames: Optional[List[str]] = None,
    append: bool = False,
    merge_mode: str = "columns",
):
    logging.debug("open_chisurf_sampling")
    if (
        filenames is None
        and getattr(ndxplorer, "_data_source", None) is not None
        and not ndxplorer._data_source.empty
    ):
        result = show_merge_dialog(ndxplorer, "Open ChiSurf Sampling Files")
        if result is None:
            return
        append, merge_mode = result
    open_files(ndxplorer, file_handles=filenames, file_type="cs_sampling", append=append, merge_mode=merge_mode)


def open_mfd_hdf5(
    ndxplorer,
    filenames: Optional[List[str]] = None,
    append: bool = False,
    merge_mode: str = "columns",
):
    logging.debug("open_mfd_hdf5")
    if (
        filenames is None
        and getattr(ndxplorer, "_data_source", None) is not None
        and not ndxplorer._data_source.empty
    ):
        result = show_merge_dialog(ndxplorer, "Open MFD HDF5 Files")
        if result is None:
            return
        append, merge_mode = result
    open_files(ndxplorer, file_handles=filenames, file_type="mfd_hdf5", append=append, merge_mode=merge_mode)


def open_smfret(ndxplorer, merge_mode: str = "columns"):
    logging.debug("open_smFRET")
    append = False
    if getattr(ndxplorer, "_data_source", None) is not None and not ndxplorer._data_source.empty:
        result = show_merge_dialog(ndxplorer, "Open SmFRET Files")
        if result is None:
            return
        append, merge_mode = result
    open_files(ndxplorer, file_type="burst_dir", append=append, merge_mode=merge_mode)
