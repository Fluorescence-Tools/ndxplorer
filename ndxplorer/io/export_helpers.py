"""Helpers for exporting/saving burst IDs and clustering data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from qtpy import QtWidgets

from ..logging_config import logging
from ..io import writer


def save_burst_ids(ndxplorer, folder: Optional[str] = None) -> None:
    """Show the burst-ID save dialog workflow."""
    logging.debug("onSaveBurstIDs")
    if folder is None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            None, "Folder for Burst IDs", ndxplorer.working_path
        )

    if not folder:
        return

    logging.info("Saving burst IDs to %s...", folder)
    writer.save_burst_ids(
        folder_name=folder,
        selections=ndxplorer.plot_control.get_selections(),
        data_source=ndxplorer.data_source,
    )

    # Record selection analysis back to database if ZMQ client is active
    if getattr(ndxplorer, "zmq_client", None) is not None and getattr(ndxplorer, "processed_data_id", None) is not None:
        try:
            selections = ndxplorer.plot_control.get_selections()
            mask = ndxplorer.data_source.get_mask(selections=selections)
            
            gate_settings = {"gate": {}}
            for sel in selections:
                gate_settings["gate"][getattr(sel, "name", "selection")] = {
                    "x_param": getattr(sel, "x_param", ""),
                    "y_param": getattr(sel, "y_param", ""),
                    "bounds": list(getattr(sel, "bounds", [])),
                }

            selection_mask_product = {
                "product_type": "selection_mask",
                "storage_mode": "embedded_json",
                "data": {"mask": mask.tolist()},
                "validation_status": "valid",
            }
            
            logging.info("Recording ndxplorer selection analysis to ChiSurf database via ZMQ...")
            res = ndxplorer.zmq_client.call(
                "ndxplorer.record_analysis",
                {
                    "experiment_id": getattr(ndxplorer, "experiment_id", None) or "exp_1",
                    "input_processed_data_ids": [ndxplorer.processed_data_id],
                    "analysis_type": "selection",
                    "settings": gate_settings,
                    "products": [selection_mask_product],
                    "software_version": "1.0.0",
                }
            )
            if res.get("ok") or (isinstance(res.get("result"), dict) and res["result"].get("ok")):
                logging.info("Successfully recorded selection analysis to database.")
            else:
                err = res.get("error", "Unknown error")
                logging.error(f"Failed to record selection analysis: {err}")
        except Exception as e:
            logging.error(f"Error recording selection analysis to database: {e}")

    dialog = QtWidgets.QDialog(ndxplorer)
    dialog.setWindowTitle("Process Burst IDs")
    layout = QtWidgets.QVBoxLayout()
    layout.addWidget(QtWidgets.QLabel("Choose what to do with the saved Burst IDs:"))

    cb_hist = QtWidgets.QCheckBox("Compute microtime histogram")
    cb_hist.setChecked(True)
    layout.addWidget(cb_hist)

    cb_correlate = QtWidgets.QCheckBox("Open FCS Correlator Wizard to correlate BST files")
    cb_correlate.setChecked(False)
    layout.addWidget(cb_correlate)

    button_box = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
    )
    button_box.accepted.connect(dialog.accept)
    button_box.rejected.connect(dialog.reject)
    layout.addWidget(button_box)
    dialog.setLayout(layout)

    if dialog.exec_() != QtWidgets.QDialog.Accepted:
        return

    do_hist = cb_hist.isChecked()
    do_corr = cb_correlate.isChecked()
    setup_name = None
    try:
        bid_folder = Path(folder)
        logging.info("Looking for setup name in %s...", bid_folder)
        info_folder = bid_folder.parent / "Info"
        if not info_folder.exists():
            info_folder = bid_folder.parent.parent / "Info"
            logging.info("Looking for setup name in %s...", info_folder)
        if info_folder.exists():
            params_file = info_folder / "photon_selection_parameters.json"
            if params_file.exists():
                with open(params_file, "r", encoding="utf-8") as handle:
                    params = json.load(handle)
                setup_name = params.get("selected_setup")
                if setup_name:
                    logging.info("Found setup name '%s' in photon_selection_parameters.json", setup_name)
    except Exception as exc:  # pragma: no cover - UI path
        logging.error("Error reading setup information: %s", exc)

    if do_hist:
        QtWidgets.QMessageBox.information(
            ndxplorer,
            "Plugin Not Available",
            "The Microtime Histogram plugin (chisurf) is not available in this standalone build.",
        )

    if do_corr:
        try:
            from chisurf.plugins.fcs.fcs_correlator.wizard import ChisurfFCSWizard

            root = Path(folder)
            bst_files = set()
            bst_files.update(str(p.resolve()) for p in root.glob("*.bst"))
            for sub in [root / "BID", root / "BID" / "ALL", root / "ALL"]:
                if sub.exists() and sub.is_dir():
                    bst_files.update(str(p.resolve()) for p in sub.glob("*.bst"))
            if not bst_files:
                bst_files.update(str(p.resolve()) for p in root.rglob("*.bst"))

            bst_files = sorted(bst_files)
            if not bst_files:
                QtWidgets.QMessageBox.information(
                    ndxplorer,
                    "No BST Files Found",
                    "No .bst files were found in the selected folder.",
                )
                return

            wiz = ChisurfFCSWizard(ndxplorer)
            ndxplorer._fcs_correlator_wizard = wiz  # keep reference
            try:
                wiz.file_page.file_list.add_files(bst_files)
                try:
                    wiz.file_page._files_or_checks_changed()
                except Exception:
                    pass
                try:
                    wiz.lineEditWorkingPath.setText(str(root))
                except Exception:
                    pass
            except Exception as exc:
                logging.debug("Preloading BST files into wizard failed: %s", exc)
            wiz.show()
            wiz.raise_()
        except Exception as exc:  # pragma: no cover - plugin path
            logging.error("Failed to launch FCS Correlator Wizard: %s", exc)
            QtWidgets.QMessageBox.warning(
                ndxplorer,
                "Launch Error",
                f"Failed to open FCS Correlator Wizard: {exc}",
            )


def save_clustering_data(ndxplorer, folder: Optional[str] = None) -> None:
    """Save clustering data using writer helper."""
    logging.debug("onSaveClusteringData")
    if ndxplorer._cluster_labels is None:
        QtWidgets.QMessageBox.warning(
            ndxplorer,
            "No Clustering Data",
            "No clustering data available. Please apply clustering before saving.",
        )
        return

    if folder is None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            None, "Folder for Clustering Data", ndxplorer.working_path
        )
    if not folder:
        return

    dialog = getattr(ndxplorer, "clustering_dialog", None)
    if dialog is None:
        QtWidgets.QMessageBox.warning(
            ndxplorer,
            "Clustering Dialog Missing",
            "Clustering dialog is not available; cannot save clustering data.",
        )
        return

    if dialog._cluster_method == "hdbscan":
        parameters = {
            "min_samples": dialog._cluster_min_samples,
            "min_cluster_size": dialog._cluster_min_cluster_size,
        }
    elif dialog._cluster_method == "kmeans":
        parameters = {"n_clusters": dialog._cluster_n_clusters}
    else:
        parameters = {}

    logging.info("Saving clustering data to %s...", folder)
    writer.save_clustering_data(
        folder_name=folder,
        data_source=ndxplorer.data_source,
        cluster_method=dialog._cluster_method,
        cluster_labels=ndxplorer._cluster_labels,
        cluster_probabilities=ndxplorer._cluster_probabilities,
        cluster_columns=dialog._cluster_columns,
        parameters=parameters,
    )

    # Record clustering analysis back to database if ZMQ client is active
    if getattr(ndxplorer, "zmq_client", None) is not None and getattr(ndxplorer, "processed_data_id", None) is not None:
        try:
            cluster_settings = {
                "method": dialog._cluster_method,
                "columns": list(dialog._cluster_columns),
                "parameters": parameters,
            }
            
            clustering_product = {
                "product_type": "clustering_labels",
                "storage_mode": "embedded_json",
                "data": {
                    "labels": ndxplorer._cluster_labels.tolist() if ndxplorer._cluster_labels is not None else [],
                    "probabilities": ndxplorer._cluster_probabilities.tolist() if ndxplorer._cluster_probabilities is not None else [],
                },
                "validation_status": "valid",
            }
            
            logging.info("Recording ndxplorer clustering analysis to ChiSurf database via ZMQ...")
            res = ndxplorer.zmq_client.call(
                "ndxplorer.record_analysis",
                {
                    "experiment_id": getattr(ndxplorer, "experiment_id", None) or "exp_1",
                    "input_processed_data_ids": [ndxplorer.processed_data_id],
                    "analysis_type": "gmm_clustering",
                    "settings": cluster_settings,
                    "products": [clustering_product],
                    "software_version": "1.0.0",
                }
            )
            if res.get("ok") or (isinstance(res.get("result"), dict) and res["result"].get("ok")):
                logging.info("Successfully recorded clustering analysis to database.")
            else:
                err = res.get("error", "Unknown error")
                logging.error(f"Failed to record clustering analysis: {err}")
        except Exception as e:
            logging.error(f"Error recording clustering analysis to database: {e}")
