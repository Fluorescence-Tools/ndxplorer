"""Colormap-related helpers extracted from plot_main."""

from __future__ import annotations

from qtpy import QtWidgets

from guiqwt.colormap import get_colormap_list

from ..logging_config import logging


def current_cmap(ndxplorer: "NDXplorer") -> str:
    return ndxplorer.comboBoxCmap.currentText()


def populate_colormap_combobox(ndxplorer: "NDXplorer") -> None:
    logging.debug("Populating colormap combobox")
    colormap_names = sorted(get_colormap_list())
    logging.debug("Found %s colormaps", len(colormap_names))
    ndxplorer.comboBoxCmap.addItems(colormap_names)

    current = current_cmap(ndxplorer)
    if current in colormap_names:
        default_index = colormap_names.index(current)
        ndxplorer.comboBoxCmap.setCurrentIndex(default_index)
        logging.info("Set default colormap to %s at index %s", current, default_index)
    else:
        logging.info("Default colormap %s not found in available colormaps", current)


def update_cmap(ndxplorer: "NDXplorer", cmap_name: str | None = None) -> None:
    if not getattr(ndxplorer, "_deferred_init_done", False) or ndxplorer.g_2dplot is None:
        return
    logging.debug("Updating colormap with cmap_name=%s", cmap_name)
    if cmap_name is None:
        cmap_name = current_cmap(ndxplorer)
        logging.debug("Using current colormap: %s", cmap_name)
    
    # Handle both backends
    if getattr(ndxplorer, '_use_simple_backend', True):
        # SimpleImageWidget uses set_colormap
        ndxplorer.cax.set_colormap(cmap_name, ndxplorer.vmin, ndxplorer.vmax)
    else:
        # guiqwt uses set_color_map
        ndxplorer.cax.set_color_map(cmap_name)
    
    ndxplorer.g_2dplot.replot()
    logging.debug("Colormap updated to %s", cmap_name)


def set_default_colormap(ndxplorer: "NDXplorer", default_cmap: str) -> None:
    logging.info("Setting default colormap to %s", default_cmap)
    index = ndxplorer.comboBoxCmap.findText(default_cmap)
    if index == -1:
        logging.info("Colormap %s not found in available colormaps", default_cmap)
        return
    ndxplorer.comboBoxCmap.setCurrentIndex(index)
    if getattr(ndxplorer, "_deferred_init_done", False) and ndxplorer.cax is not None:
        # Handle both backends
        if getattr(ndxplorer, '_use_simple_backend', True):
            # SimpleImageWidget uses set_colormap
            vmin = getattr(ndxplorer, 'vmin', 0.0)
            vmax = getattr(ndxplorer, 'vmax', 1.0)
            ndxplorer.cax.set_colormap(default_cmap, vmin, vmax)
        else:
            # guiqwt uses set_color_map
            ndxplorer.cax.set_color_map(default_cmap)
        ndxplorer.g_2dplot.replot()
    logging.info("Default colormap set to %s", default_cmap)
