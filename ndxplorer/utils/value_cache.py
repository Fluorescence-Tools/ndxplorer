"""Helpers for computing value masks and filtered data caches."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..logging_config import logging

if TYPE_CHECKING:  # pragma: no cover
    from ..core.plot_main import NDXplorer


def get_value_mask(ndxplorer: "NDXplorer") -> np.ndarray:
    """Return ndarray mask, updating caches on the ndxplorer instance."""
    selections = ndxplorer.plot_control.get_selections()
    mask_inf = ndxplorer._mask_inf
    mask_nan = ndxplorer._mask_nan
    p13 = (
        ndxplorer.plot_control.p1[0],
        ndxplorer.plot_control.p2[0],
        ndxplorer.plot_control.p3[0],
    )
    logging.debug(
        "Value mask parameters: p13=%s, mask_inf=%s, mask_nan=%s, selections=%s",
        p13,
        mask_inf,
        mask_nan,
        len(selections),
    )

    dynamic_selection = ndxplorer._dynamic_selection and hasattr(ndxplorer, "selection_z")
    selected_cluster = ndxplorer.plot_control.selected_cluster
    use_clustering = ndxplorer._use_clustering and selected_cluster >= 0

    cache_is_valid = (
        getattr(ndxplorer, "_cached_values", None) is not None
        and ndxplorer._cached_values_selections == selections
        and ndxplorer._cached_values_p13 == p13
        and ndxplorer._cached_values_mask_inf == mask_inf
        and ndxplorer._cached_values_mask_nan == mask_nan
        and getattr(ndxplorer, "_cached_values_dynamic_selection", None) == dynamic_selection
        and getattr(ndxplorer, "_cached_values_z_range", None) == getattr(ndxplorer, "_last_z_range", None)
        and getattr(ndxplorer, "_cached_values_use_clustering", None) == use_clustering
        and getattr(ndxplorer, "_cached_values_selected_cluster", None) == selected_cluster
    )
    if cache_is_valid:
        logging.debug("Using cached values")
        return ndxplorer._cached_values

    logging.debug("Cache invalid, computing fresh data")
    mask = ndxplorer.data_source.get_mask(
        selections=selections,
        idxs=[p13[0], p13[1], p13[2]],
        mask_inf=mask_inf,
        mask_nan=mask_nan,
    )

    if dynamic_selection:
        z_range = ndxplorer.selection_z.get_range()
        z_min = min(z_range)
        z_max = max(z_range)
        ndxplorer._last_z_range = z_range
        d3 = ndxplorer.data_source.values[p13[2]]
        z_select = (d3 >= z_min) & (d3 <= z_max)
        new_mask = np.zeros_like(mask)
        new_mask[:, ~z_select] = True
        mask = mask | new_mask
        logging.debug("Dynamic selection: %s points selected out of %s", np.sum(z_select), len(d3))

    if use_clustering:
        try:
            if "Cluster Label" in ndxplorer.data_source.data.columns:
                cluster_labels = ndxplorer.data_source.data["Cluster Label"].values
                cluster_mask = cluster_labels == selected_cluster
                new_mask = np.zeros_like(mask)
                new_mask[:, ~cluster_mask] = True
                mask = mask | new_mask
                points_in_cluster = np.sum(cluster_mask)
                points_in_cluster_after_masking = np.sum(~np.any(mask[:, cluster_mask], axis=0))
                logging.debug(
                    "Cluster selection: %s points in cluster %s (out of %s total in this cluster)",
                    points_in_cluster_after_masking,
                    selected_cluster,
                    points_in_cluster,
                )
            else:
                logging.warning("'Cluster Label' column not found in dataframe. Skipping cluster filtering.")
                logging.warning("This can happen if clustering has not been performed yet.")
        except Exception as exc:
            logging.warning("Error applying cluster filter: %s", exc)
            logging.warning("Skipping cluster filtering.")

    ndxplorer._cached_values_selections = selections
    ndxplorer._cached_values_p13 = p13
    ndxplorer._cached_values_mask_inf = mask_inf
    ndxplorer._cached_values_mask_nan = mask_nan
    ndxplorer._cached_values_dynamic_selection = dynamic_selection
    ndxplorer._cached_values_z_range = getattr(ndxplorer, "_last_z_range", None)
    ndxplorer._cached_values_use_clustering = use_clustering
    ndxplorer._cached_values_selected_cluster = selected_cluster
    logging.debug("Values cached for future use")
    return mask


def get_filtered_values(ndxplorer: "NDXplorer") -> np.ndarray:
    """Return filtered/cached view of ndxplorer data.
    Optimized for large datasets with memory-efficient operations."""
    if (
        hasattr(ndxplorer, "_cached_filtered_values")
        and ndxplorer._cached_filtered_values is not None
        and getattr(ndxplorer, "_cached_values_mask_id", None) == id(get_value_mask(ndxplorer))
    ):
        logging.debug("Using cached filtered values")
        return ndxplorer._cached_filtered_values

    mask = get_value_mask(ndxplorer)
    all_values = ndxplorer.data_source.values
    
    # Optimized filtering for large datasets
    # Use boolean indexing instead of masked array for better performance
    valid_mask = ~np.any(mask, axis=0)
    filtered_data = all_values[:, valid_mask]
    
    oCol, oRow = all_values.shape
    n_valid = filtered_data.shape[1]
    logging.debug("Original data shape: %sx%s, filtered to: %sx%s", oCol, oRow, oCol, n_valid)

    ndxplorer._cached_filtered_values = filtered_data
    ndxplorer._cached_values_mask_id = id(mask)
    return filtered_data
