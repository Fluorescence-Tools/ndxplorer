"""Napari integration helpers for NDxplorer."""

from __future__ import annotations

from typing import Optional, Any, Tuple

import os
import sys

from qtpy import QtWidgets

from ..logging_config import logging

_napari_module: Optional[Any] = None


def _import_napari() -> bool:
    """Attempt to import napari and cache the module."""
    global _napari_module
    if _napari_module is not None:
        return True
    try:
        import napari as _napari  # type: ignore

        _napari_module = _napari
        ver = getattr(_napari_module, "__version__", "unknown")
        loc = getattr(_napari_module, "__file__", "unknown")
        logging.debug("Imported napari library, version=%s, path=%s", ver, loc)
        return True
    except ImportError:
        _napari_module = None
        logging.debug("napari library not available (ImportError)")
        return False
    except Exception as exc:
        _napari_module = None
        logging.debug("napari import failed: %s", exc)
        return False


def is_napari_available(ndxplorer) -> bool:
    """Check if napari is available and exposes expected API."""
    if not _import_napari():
        return False
    if not hasattr(_napari_module, "Viewer"):
        ver = getattr(_napari_module, "__version__", "unknown")
        loc = getattr(_napari_module, "__file__", "unknown")
        logging.debug(
            "napari module imported but no Viewer attribute found (version=%s, path=%s). Treating as not available.",
            ver,
            loc,
        )
        _clear_cached_module()
        return False
    return True


def _clear_cached_module():
    global _napari_module
    _napari_module = None


def prompt_install_napari(ndxplorer) -> bool:
    """Prompt user via deps_installer to install napari."""
    logging.debug("prompt_install_napari")
    try:
        from .deps_installer import ensure_package_gui
    except Exception as exc:
        logging.error("deps_installer unavailable: %s", exc)
        return False

    description = (
        "Napari is not shipped with ChiSurf.\n\n"
        "napari is an open-source, multi-dimensional image viewer commonly used for scientific image analysis.\n\n"
        "Please use the Package Manager to install napari."
    )
    return ensure_package_gui(
        parent=ndxplorer,
        package="napari",
        import_name="napari",
        description=description,
        allow_pip=True,
        channels=["conda-forge", "defaults"],
    )


def install_napari_via_conda(ndxplorer) -> Tuple[bool, Optional[str]]:
    """Install napari via deps_installer conda helpers."""
    logging.info("Starting napari installation via deps_installer (conda)")
    try:
        from .deps_installer import conda_install, try_import, find_conda_executable
    except Exception as exc:
        logging.error("deps_installer not available: %s", exc)
        return False, str(exc)

    logging.info("Python: %s", sys.executable)
    logging.info("sys.prefix (target env): %s", sys.prefix)
    logging.info("CONDA_PREFIX: %s", os.environ.get("CONDA_PREFIX", ""))
    logging.info("Detected conda executable: %s", find_conda_executable())

    ok, err = conda_install(["napari"], channels=["conda-forge", "defaults"], update_deps=True)
    try:
        _ok_imp, _err_imp = try_import("napari")
        if _ok_imp:
            logging.info("napari import test succeeded after installation")
        else:
            logging.warning("napari import test failed after installation: %s", _err_imp)
    except Exception as diag_exc:
        logging.warning("Post-install diagnostics encountered an error: %s", diag_exc)
    return ok, err


def ensure_napari_available(ndxplorer: "NDXplorer") -> bool:
    """Ensure napari is importable, prompting install if necessary."""
    if is_napari_available(ndxplorer):
        return True
    installed = prompt_install_napari(ndxplorer)
    if not installed:
        return False
    _clear_cached_module()
    if not is_napari_available(ndxplorer):
        QtWidgets.QMessageBox.warning(
            ndxplorer,
            "Napari Not Available",
            "napari could not be loaded even after installation. Please restart ChiSurf and try again.",
        )
        return False
    return True


def send_to_napari(ndxplorer: "NDXplorer") -> None:
    """Send current 2D histogram to napari, prompting install if needed."""
    logging.debug("send_to_napari")
    if not ensure_napari_available(ndxplorer):
        return

    if not hasattr(ndxplorer, "_histogram") or "2d" not in ndxplorer._histogram:
        logging.warning("No 2D histogram data available to send to napari")
        return
    
    H, x_edges, y_edges = ndxplorer._histogram["2d"]
    hist_data = H.T
    x_label = getattr(ndxplorer.plot_control, "x_label", "X")
    y_label = getattr(ndxplorer.plot_control, "y_label", "Y")
    weight_label = ndxplorer.plot_control.weight_parameter

    viewer = get_or_create_viewer(ndxplorer)
    if viewer is None:
        return

    viewer.add_image(
        hist_data,
        name=f"NDXplorer: {x_label} vs {y_label} {weight_label}",
        colormap="viridis",
        scale=[1, 1],
    )
    logging.info("Sent 2D histogram to napari: %s vs %s", x_label, y_label)


def get_or_create_viewer(ndxplorer) -> Optional["napari.Viewer"]:
    napari = _napari_module
    if napari is None:
        return None
    viewer = None
    try:
        current_viewer_fn = getattr(napari, "current_viewer", None)
        if callable(current_viewer_fn):
            viewer = current_viewer_fn()
    except Exception as e:
        logging.debug("napari.current_viewer() not usable: %s", e)

    if viewer is not None:
        return viewer

    if not hasattr(napari, "Viewer"):
        QtWidgets.QMessageBox.warning(
            ndxplorer,
            "Napari Viewer Unavailable",
            "The installed napari does not expose a Viewer API compatible with this feature.\n"
            "Please update napari (e.g., via conda-forge) and try again.",
        )
        return None

    try:
        return napari.Viewer()
    except Exception as e:
        logging.warning("Failed to create napari.Viewer(): %s", e)
        QtWidgets.QMessageBox.critical(
            ndxplorer,
            "Could not open Napari",
            f"Napari is installed, but the viewer could not be created.\n\nDetails:\n{e}",
        )
        return None


# Backwards compatibility for any legacy imports
_get_or_create_viewer = get_or_create_viewer
