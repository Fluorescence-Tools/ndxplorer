"""Utilities for managing the clustering dialog, column selection, and worker lifecycle."""

from __future__ import annotations

from typing import Optional, Set, Dict, Any, Tuple

import numpy as np
from qtpy import QtWidgets

from ..logging_config import logging
from ..ui.clustering_dialog import ClusteringDialog
from ..ui.column_selection_dialog import ColumnSelectionDialog
from ..analysis.clustering import ClusteringManager, ClusteringWorker
from ..utils.lazy_imports import get_hdbscan, get_kmeans

if False:  # pragma: no cover
    from ..core.plot_main import NDXplorer


def setup_button(ndxplorer: "NDXplorer") -> None:
    """Wire the Clustering button to show the dialog."""
    ndxplorer.pushButtonShowClusteringDialog.clicked.connect(
        lambda: toggle_dialog(ndxplorer)
    )


def ensure_dialog(ndxplorer: "NDXplorer") -> None:
    """Create the clustering dialog if missing and wire signals."""
    if getattr(ndxplorer, "clustering_dialog", None) is not None:
        return

    dialog = ClusteringDialog(parent=ndxplorer)
    ndxplorer.clustering_dialog = dialog
    update_dialog(ndxplorer, force=True)
    dialog.clustering_done.connect(ndxplorer.on_clustering_done)
    dialog.clustering_error.connect(ndxplorer.on_clustering_error)
    dialog.progress_updated.connect(ndxplorer.on_clustering_progress)


def toggle_dialog(ndxplorer: "NDXplorer") -> None:
    """Show or hide the clustering dialog."""
    ensure_dialog(ndxplorer)
    dialog = ndxplorer.clustering_dialog
    if dialog.isVisible():
        dialog.hide()
    else:
        update_dialog(ndxplorer)
        dialog.show()


def update_dialog(ndxplorer: "NDXplorer", force: bool = False) -> None:
    """Update dialog labels/buttons with latest state."""
    dialog = getattr(ndxplorer, "clustering_dialog", None)
    if dialog is None:
        return
    if not force and not dialog.isVisible():
        return
    try:
        num_selected = len(dialog._cluster_columns)
        if num_selected > 0:
            dialog.pushButtonSelectColumns.setText(
                f"Select Columns (Recommended) ({num_selected})"
            )
        else:
            dialog.pushButtonSelectColumns.setText("Select Columns (Recommended)")
        dialog.pushButtonSaveClustering.setEnabled(
            getattr(ndxplorer, "_cluster_labels", None) is not None
        )
    except RuntimeError:
        logging.warning("Clustering dialog widgets unavailable; recreating dialog.")
        ndxplorer.clustering_dialog = None
        ensure_dialog(ndxplorer)


def select_columns(ndxplorer: "NDXplorer") -> None:
    """Open the column selection dialog (from either parent or embedded dialog)."""
    ensure_dialog(ndxplorer)
    dialog = ndxplorer.clustering_dialog
    if dialog.isVisible():
        dialog.on_select_columns()
        return

    parameter_names = ndxplorer.data_source.parameter_names
    selection_dialog = ColumnSelectionDialog(
        parent=ndxplorer,
        column_names=parameter_names,
        selected_columns=dialog._cluster_columns,
    )
    if selection_dialog.exec_():
        dialog._cluster_columns = selection_dialog.get_selected_columns()
        update_dialog(ndxplorer, force=True)


def start_clustering_from_dialog(
    ndxplorer: "NDXplorer", method: str, columns: Set[str], params: Dict[str, Any]
) -> None:
    """Apply dialog-driven settings then defer to apply_clustering."""
    ensure_dialog(ndxplorer)
    dialog = ndxplorer.clustering_dialog
    dialog._cluster_method = method
    dialog._cluster_columns = columns
    apply_clustering(ndxplorer)


def _ensure_algorithm_available(ndxplorer: "NDXplorer", method: str) -> bool:
    """Check/import clustering dependency, optionally prompting install."""
    if method == "hdbscan":
        if get_hdbscan() is not None:
            return True
        try:
            from .deps_installer import ensure_package_gui

            desc = (
                "HDBSCAN is a density-based clustering algorithm useful "
                "for finding clusters of varying densities and shapes."
            )
            installed = ensure_package_gui(
                parent=ndxplorer,
                package="hdbscan",
                import_name="hdbscan",
                description=desc,
                allow_pip=True,
                channels=["conda-forge", "defaults"],
            )
        except Exception as exc:  # pragma: no cover - UI flow
            logging.warning("Could not run installer for hdbscan: %s", exc)
            installed = False
        if not installed:
            return False
        if get_hdbscan() is None:
            QtWidgets.QMessageBox.information(
                ndxplorer,
                "HDBSCAN Installed",
                "HDBSCAN was installed but could not be imported immediately.\n"
                "Please restart ChiSurf and try again.",
            )
            return False
        return True

    if method == "kmeans":
        if get_kmeans() is not None:
            return True
        QtWidgets.QMessageBox.warning(
            ndxplorer,
            "scikit-learn Not Available",
            "scikit-learn is not installed. Please install it using pip or conda.",
        )
        return False

    return True


def apply_clustering(ndxplorer: "NDXplorer") -> None:
    """Start clustering worker using current dialog selections."""
    ensure_dialog(ndxplorer)
    ndxplorer._use_clustering = True
    dialog = ndxplorer.clustering_dialog
    method = dialog._cluster_method
    columns = dialog._cluster_columns

    if not _ensure_algorithm_available(ndxplorer, method):
        if dialog is not None:
            dialog.checkBoxClustering.setChecked(False)
        return

    params: Dict[str, Any]
    if method == "hdbscan":
        params = {
            "min_samples": dialog._cluster_min_samples,
            "min_cluster_size": dialog._cluster_min_cluster_size,
        }
    elif method == "kmeans":
        params = {"n_clusters": dialog._cluster_n_clusters}
    else:
        params = {}

    worker = getattr(ndxplorer, "clustering_worker", None)
    if worker is not None:
        try:
            worker.clustering_done.disconnect(ndxplorer.on_clustering_done)
            worker.clustering_error.disconnect(ndxplorer.on_clustering_error)
            worker.progress_updated.disconnect(ndxplorer.on_clustering_progress)
        except (TypeError, RuntimeError):
            pass
        if worker.isRunning():
            worker.wait()
        worker.deleteLater()

    ndxplorer.clustering_worker = ClusteringWorker(ndxplorer, method, params)
    ndxplorer.clustering_worker.clustering_done.connect(ndxplorer.on_clustering_done)
    ndxplorer.clustering_worker.clustering_error.connect(
        ndxplorer.on_clustering_error
    )
    ndxplorer.clustering_worker.progress_updated.connect(
        ndxplorer.on_clustering_progress
    )
    ndxplorer.clustering_worker.start()


def cancel_clustering(ndxplorer: "NDXplorer") -> None:
    """Request the worker to stop and update dialog state."""
    worker = getattr(ndxplorer, "clustering_worker", None)
    if worker is not None and worker.isRunning():
        worker.stop()
        dialog = getattr(ndxplorer, "clustering_dialog", None)
        if dialog is not None and dialog.isVisible():
            dialog.pushButtonCancelClustering.setText("Cancelling...")
            dialog.pushButtonCancelClustering.setEnabled(False)


def perform_clustering(
    ndxplorer: "NDXplorer",
    method: Optional[str] = None,
    worker=None,
    **kwargs,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Execute clustering via ClusteringManager and update ndxplorer state.
    """
    if not hasattr(ndxplorer, "_clustering_manager"):
        ndxplorer._clustering_manager = ClusteringManager(ndxplorer._data_source)
    else:
        ndxplorer._clustering_manager._data_source = ndxplorer._data_source

    if method is None:
        method = ndxplorer.clustering_dialog._cluster_method

    if method == "hdbscan":
        min_samples = kwargs.get(
            "min_samples", ndxplorer.clustering_dialog._cluster_min_samples
        )
        min_cluster_size = kwargs.get(
            "min_cluster_size", ndxplorer.clustering_dialog._cluster_min_cluster_size
        )
        kwargs["min_samples"] = min_samples
        kwargs["min_cluster_size"] = min_cluster_size
    elif method == "kmeans":
        n_clusters = kwargs.get(
            "n_clusters", ndxplorer.clustering_dialog._cluster_n_clusters
        )
        kwargs["n_clusters"] = n_clusters

    if ndxplorer.clustering_dialog._cluster_columns:
        kwargs["columns"] = ndxplorer.clustering_dialog._cluster_columns
    else:
        kwargs["x_values"] = ndxplorer.x_values
        kwargs["y_values"] = ndxplorer.y_values
        kwargs["z_values"] = ndxplorer.z_values

    result = ndxplorer._clustering_manager.perform_clustering(
        method=method, worker=worker, **kwargs
    )

    if result[0] is not None and result[1] is not None:
        full_labels, full_probabilities = result
        ndxplorer._cluster_labels = full_labels
        ndxplorer._cluster_probabilities = full_probabilities
        logging.info("Stored cluster labels and probabilities to prevent infinite loop")

        df = ndxplorer._data_source.data
        df["Cluster Label"] = full_labels
        df["Cluster Probability"] = full_probabilities
        ndxplorer._data_source.data = df

        ndxplorer.plot_control.update(update_comboboxes=True, update_plots=False)
        logging.info("Refreshed axis selection comboboxes to show cluster columns")

        ndxplorer._cluster_data_shape = (
            ndxplorer._clustering_manager._cluster_data_shape
        )
        logging.info("Stored cluster data shape: %s", ndxplorer._cluster_data_shape)

    return result

