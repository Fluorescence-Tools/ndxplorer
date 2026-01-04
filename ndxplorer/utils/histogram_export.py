"""Utilities for exporting histogram data to clipboard."""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING

import numpy as np
from qtpy import QtWidgets

from ..logging_config import logging

if TYPE_CHECKING:  # pragma: no cover
    from ..core.plot_main import NDXplorer


def _fmt_num(v: float) -> str:
    """Format numbers with high precision; use scientific for very small/large."""
    try:
        if isinstance(v, (int, np.integer)):
            return f"{int(v)}"
        if np.isfinite(v) and np.isclose(v, round(v), rtol=0.0, atol=1e-12):
            return f"{int(round(v))}"
        return f"{float(v):.8g}"
    except Exception:
        return str(v)


def copy_1d_histograms(ndxplorer: "NDXplorer") -> None:
    """Copy X/Y 1D histogram data as TSV to clipboard."""
    logging.debug("copy_1d_hists_to_clipboard_csv")
    try:
        x_hist = ndxplorer._histogram["x"]
        y_hist = ndxplorer._histogram["y"]
    except KeyError as exc:
        logging.error("Histogram data missing: %s", exc)
        return

    output = io.StringIO()
    output.write("Histogram\tBinCenter\tCount\n")

    for label, hist in (("X", x_hist), ("Y", y_hist)):
        bin_edges, counts = hist
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        for center, count in zip(bin_centers, counts):
            output.write(f"{label}\t{_fmt_num(center)}\t{_fmt_num(count)}\n")

    QtWidgets.QApplication.clipboard().setText(output.getvalue())
    logging.info("1D histograms copied to clipboard as TSV.")


def copy_2d_hist_json(ndxplorer: "NDXplorer") -> None:
    """Copy 2D histogram data as JSON."""
    logging.debug("copy_2d_hist_to_clipboard_json")
    try:
        H, x_edges, y_edges = ndxplorer._histogram["2d"]
    except KeyError as exc:
        logging.error("No 2D histogram data available: %s", exc)
        return

    data_dict = {
        "H": H.tolist(),
        "x_edges": x_edges.tolist(),
        "y_edges": y_edges.tolist(),
    }
    QtWidgets.QApplication.clipboard().setText(json.dumps(data_dict, indent=2))
    logging.info("2D histogram data copied to clipboard (JSON).")


def copy_2d_hist_csv(ndxplorer: "NDXplorer") -> None:
    """Copy 2D histogram data as TSV."""
    logging.debug("copy_2d_hist_to_clipboard_csv: %s", ndxplorer._histogram)
    try:
        H, x_edges, y_edges = ndxplorer._histogram["2d"]
    except KeyError as exc:
        logging.error("No 2D histogram data available: %s", exc)
        return

    # Check if H is 2D, if not, we have a problem
    if not hasattr(H, 'shape') or len(H.shape) != 2:
        logging.error("H is not 2D! Shape: %s, Type: %s", 
                     getattr(H, 'shape', 'no shape'), type(H))
        return

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    output = io.StringIO()
    header_cells = ["y/x"] + [_fmt_num(x) for x in x_centers]
    output.write("\t".join(header_cells) + "\n")

    for j, y in enumerate(y_centers):
        row_cells = [_fmt_num(y)]
        for i in range(len(x_centers)):
            row_cells.append(_fmt_num(H[i, j]))
        output.write("\t".join(row_cells) + "\n")

    QtWidgets.QApplication.clipboard().setText(output.getvalue())
    logging.info("2D histogram data copied to clipboard as TSV.")
