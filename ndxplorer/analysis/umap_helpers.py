"""Helper functions for UMAP-related workflows (column computation & plotting)."""

from __future__ import annotations

from typing import Dict, Set

import numpy as np
import pandas as pd
from qtpy import QtWidgets

from ..logging_config import logging
from ..utils.lazy_imports import get_umap
from ..plotting import plot_umap

if False:  # pragma: no cover
    from ..core.plot_main import NDXplorer


def ensure_umap_available(parent) -> bool:
    """Check/optionally install umap-learn."""
    if get_umap() is not None:
        return True
    try:
        from ..deps_installer import ensure_package_gui

        desc = (
            "UMAP (Uniform Manifold Approximation and Projection) is a dimensionality "
            "reduction technique used to project high-dimensional data into 2D/3D for visualization."
        )
        installed = ensure_package_gui(
            parent=parent,
            package="umap-learn",
            import_name="umap",
            description=desc,
            allow_pip=True,
            channels=["conda-forge", "defaults"],
        )
    except Exception as exc:  # pragma: no cover - UI path
        installed = False
        logging.warning("Could not run installer for umap-learn: %s", exc)
    if not installed:
        return False
    if get_umap() is None:
        QtWidgets.QMessageBox.information(
            parent,
            "UMAP Installed",
            "UMAP (umap-learn) was installed but could not be imported immediately.\n"
            "Please restart ChiSurf and try again.",
        )
        return False
    return True


def compute_umap_embedding(clean_data: np.ndarray, params: Dict[str, float]):
    """Run UMAP with provided parameters and return embedding."""
    reducer = get_umap().UMAP(**params)
    return reducer.fit_transform(clean_data)


def prepare_umap_params(params: Dict[str, float]) -> Dict[str, float]:
    """Normalize param dict to defaults expected by UMAP worker."""
    umap_params = {
        "n_neighbors": params.get("n_neighbors", 15),
        "min_dist": params.get("min_dist", 0.1),
        "n_components": params.get("n_components", 2),
        "n_jobs": params.get("n_jobs", -1),
        "verbose": True,
        "tqdm_kwds": {"desc": "UMAP Embedding", "unit": "epoch"},
    }
    for name in [
        "metric",
        "learning_rate",
        "init",
        "spread",
        "low_memory",
        "set_op_mix_ratio",
        "local_connectivity",
        "repulsion_strength",
        "negative_sample_rate",
        "n_epochs",
    ]:
        if name in params:
            umap_params[name] = params[name]
    if umap_params["n_jobs"] == 1:
        umap_params["random_state"] = 42
    return umap_params


def add_umap_columns(
    ndxplorer: "NDXplorer",
    columns: Set[str],
    params: Dict[str, float],
    progress_dialog_factory,
) -> bool:
    """Compute UMAP columns and add them to the NDXplorer data source."""
    logging.info("Adding UMAP columns using: %s", columns)
    if not ensure_umap_available(ndxplorer):
        return False
    if ndxplorer._data_source is None or ndxplorer._data_source.empty:
        logging.error("No data available for UMAP transformation.")
        return False

    df = ndxplorer._data_source.data
    selected_data = []
    for column in columns:
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce").values
            selected_data.append(values)
    if not selected_data:
        logging.error("No valid columns found for UMAP transformation.")
        return False

    data = np.column_stack(selected_data)
    mask = ~np.any(np.isnan(data) | np.isinf(data), axis=1)
    clean_data = data[mask]
    umap_params = prepare_umap_params(params)
    if len(clean_data) < umap_params["n_neighbors"]:
        logging.warning(
            "Not enough data points for UMAP. Need at least %d.", umap_params["n_neighbors"]
        )
        return False

    progress_dialog = progress_dialog_factory("UMAP Column Computation")
    progress_dialog.run_umap_computation(clean_data, umap_params)
    progress_dialog.exec_()
    umap_embedding = progress_dialog.get_result()
    error_message = progress_dialog.get_error()
    if error_message:
        logging.error("Error during UMAP transformation: %s", error_message)
        return False
    if umap_embedding is None:
        logging.error("UMAP computation was cancelled or failed")
        return False

    projections = []
    for i in range(umap_params["n_components"]):
        full_projection = np.full(len(data), np.nan)
        full_projection[mask] = umap_embedding[:, i]
        projections.append(full_projection)

    for i, projection in enumerate(projections):
        column_name = f"UMAP_{i+1}"
        df[column_name] = projection
        logging.info("Added column '%s'", column_name)

    ndxplorer._data_source.data = df
    ndxplorer.refresh_axis_comboboxes_preserving_selection()
    return True


def create_umap_plot(
    ndxplorer: "NDXplorer",
    columns: Set[str],
    params: Dict[str, float],
) -> None:
    """Delegate to plot_umap module to render a UMAP plot."""
    cluster_labels = getattr(ndxplorer, "_cluster_labels", None)
    plot_umap.create_umap_plot(
        parent=ndxplorer,
        columns=columns,
        params=params,
        data_source=ndxplorer._data_source,
        x_values=ndxplorer.x_values,
        y_values=ndxplorer.y_values,
        z_values=ndxplorer.z_values,
        cluster_labels=cluster_labels,
    )
