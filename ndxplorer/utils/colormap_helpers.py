"""Colormap-related helpers extracted from plot_main."""

from __future__ import annotations

from ..logging_config import logging
from ..plotting import colormaps


def current_cmap(ndxplorer: "NDXplorer") -> str:
    return ndxplorer.comboBoxCmap.currentText()


def populate_colormap_combobox(ndxplorer: "NDXplorer") -> None:
    logging.debug("Populating colormap combobox")
    colormap_names = colormaps.get_available_colormaps()
    logging.debug("Found %s colormaps", len(colormap_names))
    ndxplorer.comboBoxCmap.clear()
    ndxplorer.comboBoxCmap.addItems(colormap_names)

    current = current_cmap(ndxplorer)
    if current in colormap_names:
        default_index = colormap_names.index(current)
        ndxplorer.comboBoxCmap.setCurrentIndex(default_index)
        logging.info("Set default colormap to %s at index %s", current, default_index)
    else:
        logging.info("Default colormap %s not found in available colormaps", current)


def update_cmap(ndxplorer: "NDXplorer", cmap_name: str | None = None) -> None:
    colormaps.update_colormap(ndxplorer, cmap_name)


def set_default_colormap(ndxplorer: "NDXplorer", default_cmap: str) -> None:
    colormaps.set_default_colormap(ndxplorer, default_cmap)
