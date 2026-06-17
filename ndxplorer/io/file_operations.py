"""Utility helpers to keep the file-loading logic for NDXplorer separate from plot_main."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from qtpy import QtWidgets

from ..logging_config import logging
from ..io import reader

if False:  # pragma: no cover - circular import safety for type checkers
    from .plot_main import NDXplorer


def _ensure_sequence(handles: Optional[Sequence[str]]) -> Sequence[str]:
    """Normalize Qt's return types (tuple/list) to a plain tuple."""
    if handles is None or isinstance(handles, bool):
        return ()
    if isinstance(handles, (list, tuple)):
        return tuple(handles)
    return (handles,)


def _update_working_path(ndxplorer: "NDXplorer", first_selection: Optional[str]) -> None:
    """Keep ndxplorer.working_path synced with the most recent selection."""
    if not first_selection:
        return
    try:
        dir_path = str(Path(first_selection).parent)
    except Exception as exc:  # pragma: no cover - defensive
        logging.debug("Could not derive working path: %s", exc)
        return
    ndxplorer.working_path = dir_path


def _handle_append(ndxplorer: "NDXplorer", new_source, merge_mode: str) -> None:
    """Append/replace loaded data followed by UI updates."""
    if (
        hasattr(ndxplorer, "_data_source")
        and ndxplorer._data_source is not None
        and not ndxplorer._data_source.empty
    ):
        if ndxplorer._data_source.merge(new_source, mode=merge_mode):
            ndxplorer.update()
    else:
        ndxplorer._data_source = new_source
        ndxplorer.update()


def open_files(
    ndxplorer: "NDXplorer",
    file_handles: Optional[Sequence[str]] = None,
    file_type: Optional[str] = None,
    append: bool = False,
    merge_mode: str = "columns",
) -> None:
    """Central entry point for all data-loading actions."""
    logging.info("NDXplorer: Opening files..")
    logging.debug("Merge mode: %s", merge_mode)
    
    # Auto-detect sampling folder
    file_handles_seq = _ensure_sequence(file_handles)
    if not file_type and file_handles_seq and len(file_handles_seq) == 1:
        p = Path(file_handles_seq[0])
        if p.is_dir() and (p / "parameters.json").exists():
            file_type = "sampling_folder"

    file_handles_seq = _ensure_sequence(file_handles)
    reader_input = file_handles_seq
    working_path = str(ndxplorer.working_path)

    # --- Sampling / ER4 ----------------------------------------------------
    if file_type == "cs_sampling":
        if not file_handles_seq:
            directory = QtWidgets.QFileDialog.getExistingDirectory(
                ndxplorer, "Open sampling folder", working_path
            )
            file_handles_seq = (directory,) if directory else ()
        
        if file_handles_seq:
            reader_input = str(file_handles_seq[0])
            logging.info("Opening sampling folder: %s", reader_input)
            data_reader = reader.read_sampling_folder
            _update_working_path(ndxplorer, reader_input)
        else:
            reader_input = ()

    elif file_type == "er4":
        if not file_handles_seq:
            file_handles_seq, _ = QtWidgets.QFileDialog.getOpenFileNames(
                ndxplorer,
                "ChiSurf sampling files",
                working_path,
                "Sampling files (*.er4);;All files (*.*)",
            )
        if file_handles_seq:
            _update_working_path(ndxplorer, file_handles_seq[0])
            logging.info("Opening files (%s): %s", file_type, file_handles_seq)
            data_reader = reader.read_csv_sampling
            reader_input = file_handles_seq
        else:
            reader_input = ()

    # --- Sampling Folder (New Format) --------------------------------------
    elif file_type == "sampling_folder":
        if not file_handles_seq:
            directory = QtWidgets.QFileDialog.getExistingDirectory(
                ndxplorer, "Open sampling folder", working_path
            )
            file_handles_seq = (directory,) if directory else ()
        
        if file_handles_seq:
            reader_input = file_handles_seq[0]
            data_reader = reader.read_sampling_folder
        else:
            reader_input = ()

    # --- HDF5 / zipped HDF5 ------------------------------------------------
    elif file_type == "mfd_hdf5":
        if not file_handles_seq:
            file_handles_seq, _ = QtWidgets.QFileDialog.getOpenFileNames(
                ndxplorer,
                "MFD HDF5 files",
                working_path,
                "HDF5 files (*.h5 *.hdf5);;ZIP files (*.zip);;All Files (*.*)",
            )

        _update_working_path(ndxplorer, file_handles_seq[0] if file_handles_seq else None)
        logging.info("Opening MFD HDF5/Zip files: %s", file_handles_seq)

        if not file_handles_seq:
            return

        combined_data_source = None
        for file_path in file_handles_seq:
            path_str = str(file_path)
            is_zip = path_str.lower().endswith(".zip")

            if is_zip:
                try:
                    import zipfile as _zip

                    with _zip.ZipFile(path_str, "r") as zf:
                        names = zf.namelist()
                    has_h5 = any(
                        name.lower().endswith((".h5", ".hdf5")) for name in names
                    )
                    logging.debug("ZIP '%s' contains HDF5: %s", path_str, has_h5)
                except Exception as exc:
                    logging.debug("Could not inspect zip '%s': %s", path_str, exc)
                    has_h5 = False

                temp_ds = (
                    reader.read_mfd_hdf5([path_str])
                    if has_h5
                    else reader.read_burst_analysis(path_str)
                )
            else:
                temp_ds = reader.read_mfd_hdf5([path_str])

            if combined_data_source is None:
                combined_data_source = temp_ds
            else:
                combined_data_source.merge(temp_ds, mode=merge_mode)

        if append:
            _handle_append(ndxplorer, combined_data_source, merge_mode)
        else:
            ndxplorer._data_source = combined_data_source
            ndxplorer.update()

        _apply_axes_and_refresh(ndxplorer)
        logging.debug("Handled mfd_hdf5; returning before generic loader.")
        return

    # --- Generic CSV loader ------------------------------------------------
    else:
        if not file_handles_seq:
            file_handles_seq, _ = QtWidgets.QFileDialog.getOpenFileNames(
                ndxplorer,
                "Comma separated value files",
                working_path,
                "Text files (*.csv *.dat *.er4 *.txt);;All files (*.*)",
            )
        _update_working_path(ndxplorer, file_handles_seq[0] if file_handles_seq else None)
        
        # Auto-detect .er4 in generic loader
        if any(str(f).lower().endswith(".er4") for f in file_handles_seq):
            data_reader = reader.read_csv_sampling
        else:
            data_reader = reader.read_csv
        reader_input = file_handles_seq

    if reader_input:
        logging.info("Opening files (%s): %s", file_type or "csv", file_handles_seq)

        # Capture equations and constants to use in worker thread
        # Important: do this BEFORE launching the thread to avoid main-thread access issues
        equations = getattr(ndxplorer, "equations", [])
        constants = getattr(ndxplorer, "constants", {})

        def load_callable():
            # Perform initial raw data load
            ds = data_reader(reader_input)
            
            # If successful and not empty, perform heavy column computations in background
            if ds and not ds.empty:
                logging.info(f"Background: Computing columns for {ds.size} rows")
                ds.compute_columns(constants=constants, equations=equations)
                logging.info("Background: Column computation complete.")
            return ds

        _dispatch_data_load(
            ndxplorer,
            f"Loading {file_type or 'CSV'} files",
            load_callable,
            append,
            merge_mode,
        )


def _dispatch_data_load(
    ndxplorer: "NDXplorer",
    description: str,
    load_callable,
    append: bool,
    merge_mode: str,
) -> None:
    """Route loading through NDxplorer's async runner when available."""
    runner = getattr(ndxplorer, "_run_data_load_task", None)
    if callable(runner):
        runner(description, load_callable, append, merge_mode)
    else:
        data_source = load_callable()
        _finalize_loaded_data(ndxplorer, data_source, append, merge_mode)


def _finalize_loaded_data(
    ndxplorer: "NDXplorer", data_source, append: bool, merge_mode: str
) -> None:
    """Apply the loaded data to NDxplorer and refresh the UI."""
    logging.debug("_finalize_loaded_data")
    
    # Block histogram computations during initial data loading
    # This prevents computing histograms with wrong bins before image detection
    if hasattr(ndxplorer, 'plot_control'):
        ndxplorer.plot_control._loading_data = True
        logging.info("Set _loading_data flag to block premature histogram computation")
    
    if data_source is None:
        return
    
    # IMPORTANT: Detect image axes BEFORE setting data_source
    # The data_source setter calls compute_columns() which may replace raw columns
    # So we need to check for X pixel/Y pixel in the RAW loaded data first
    has_image_data = False
    image_dims = None
    if not append and data_source is not None and not data_source.empty:
        try:
            param_names = list(data_source.parameter_names)
            logging.info(f"Raw data has {len(param_names)} parameters before compute_columns: {param_names[:10]}")
            
            has_x_pixel = any("x pixel" in name.lower() for name in param_names)
            has_y_pixel = any("y pixel" in name.lower() for name in param_names)
            
            if has_x_pixel and has_y_pixel:
                has_image_data = True
                x_pixel_param = next((name for name in param_names if "x pixel" in name.lower()), None)
                y_pixel_param = next((name for name in param_names if "y pixel" in name.lower()), None)
                
                # Get image dimensions from raw data
                x_values = data_source.values[param_names.index(x_pixel_param), :]
                y_values = data_source.values[param_names.index(y_pixel_param), :]
                
                x_pixels = int(np.max(x_values)) + 1 if len(x_values) > 0 else 256
                y_pixels = int(np.max(y_values)) + 1 if len(y_values) > 0 else 256
                
                image_dims = (x_pixels, y_pixels, x_pixel_param, y_pixel_param)
                logging.info(f"Detected image data in raw loaded data: {x_pixels}x{y_pixels} pixels")
        except Exception as e:
            logging.error(f"Error detecting image data in raw data: {e}")
    
    if append:
        _handle_append(ndxplorer, data_source, merge_mode)
    else:
        # For image data, skip compute_columns to preserve raw pixel columns
        if has_image_data:
            logging.info("Skipping compute_columns for image data to preserve pixel columns")
            # Bypass BOTH the setter AND data_manager to preserve raw columns
            # Directly set the internal _data_source attribute
            object.__setattr__(ndxplorer, '_data_source', data_source)
            ndxplorer.invalidate_values_cache()
            ndxplorer._set_data_loaded(not data_source.empty)
            # Also bypass data_manager if it exists
            if hasattr(ndxplorer, 'data_manager'):
                object.__setattr__(ndxplorer.data_manager, '_data_source', data_source)
            # Enable UI controls now that data is loaded
            ndxplorer.update_ui_enabled_state()
            logging.info(f"Image data loaded with {len(data_source.parameter_names)} raw parameters preserved")
        else:
            # Normal data: use the setter which calls compute_columns
            ndxplorer.data_source = data_source
            ndxplorer.update_ui_data()
        
        # Apply image settings if detected in raw data
        if has_image_data and image_dims:
            x_pixels, y_pixels, x_pixel_param, y_pixel_param = image_dims
            logging.info(f"Applying image settings: {x_pixels}x{y_pixels}, axes: {x_pixel_param}, {y_pixel_param}")
            
            # Store image info for deferred axis detection
            ndxplorer._detected_image_dims = image_dims
        # Populate comboboxes with BOTH raw AND computed parameter names
        # This is critical so X pixel/Y pixel are available for selection
        raw_param_names = []
        if has_image_data and image_dims:
            # Get raw parameter names from the original data_source before compute_columns
            try:
                raw_param_names = list(data_source.parameter_names)
                logging.info(f"Including {len(raw_param_names)} raw parameter names in combo boxes")
            except Exception:
                pass
        
        # After compute_columns, get the computed parameter names from the public data_source
        active_ds = ndxplorer.data_source
        computed_param_names = list(active_ds.parameter_names) if active_ds and not active_ds.empty else []
        
        # Combine raw and computed, preserving order and removing duplicates
        all_param_names = []
        seen = set()
        for name in raw_param_names + computed_param_names:
            if name not in seen:
                all_param_names.append(name)
                seen.add(name)
        
        # Only perform manual population if we actually have parameters to show,
        # or if we are in the image data path where we skip the full update().
        if all_param_names:
            logging.info(f"Populating combo boxes with {len(all_param_names)} total parameters")
            
            for combo in [ndxplorer.plot_control.comboBoxSelX, 
                          ndxplorer.plot_control.comboBoxSelY,
                          ndxplorer.plot_control.comboBoxSelZ]:
                combo.blockSignals(True)
                combo.clear()
                combo.addItems(all_param_names)
                combo.blockSignals(False)
            if hasattr(ndxplorer.plot_control, 'comboBoxWeight'):
                ndxplorer.plot_control.comboBoxWeight.blockSignals(True)
        else:
            logging.warning("No parameters found to populate combo boxes.")
    _apply_axes_and_refresh(ndxplorer, all_param_names)


def _apply_axes_and_refresh(ndxplorer: "NDXplorer", all_param_names: List[str]) -> None:
    """Shared tail for open operations: apply axes + refresh plots."""
    # Defer axis detection until AFTER comboboxes are fully populated
    # (plot_control.update() may take time to restore selections)
    from qtpy.QtCore import QTimer
    
    def _deferred_axis_detection():
        """Run after combobox update completes."""
        logging.info("Running deferred image axis detection")
        
        # Check if current axes are valid for the new data
        # If not, pick sensible defaults
        p1_name = ndxplorer.plot_control.comboBoxSelX.currentText()
        p2_name = ndxplorer.plot_control.comboBoxSelY.currentText()
        
        needs_auto_axis = False
        if p1_name not in all_param_names or p2_name not in all_param_names:
            needs_auto_axis = True
            logging.info(f"Current axes ({p1_name}, {p2_name}) not in new data. Selecting defaults.")
            
        # Check if image dimensions were detected in raw data before compute_columns
        detected_dims = getattr(ndxplorer, '_detected_image_dims', None)
        image_axes_already_set = False
        
        if detected_dims:
            x_pixels, y_pixels, x_pixel_param, y_pixel_param = detected_dims
            logging.info(f"Using pre-detected image dimensions: {x_pixels}x{y_pixels}")
            
            # Set bins
            if hasattr(ndxplorer, 'plot_control'):
                ndxplorer.plot_control.n_xhist_1d = x_pixels
                ndxplorer.plot_control.n_yhist_1d = y_pixels
                ndxplorer.plot_control.n_xhist_2d = x_pixels
                ndxplorer.plot_control.n_yhist_2d = y_pixels
                
                # Update UI spinboxes
                if hasattr(ndxplorer.plot_control, 'spinBoxNXHist2D'):
                    ndxplorer.plot_control.spinBoxNXHist2D.setValue(x_pixels)
                if hasattr(ndxplorer.plot_control, 'spinBoxNYHist2D'):
                    ndxplorer.plot_control.spinBoxNYHist2D.setValue(y_pixels)
                
                # Set axes to X pixel and Y pixel
                logging.info(f"Setting axes to image parameters: X={x_pixel_param}, Y={y_pixel_param}")
                x_set = ndxplorer.plot_control.set_axis_by_name("x", x_pixel_param, match_contains=False, block_signals=False)
                y_set = ndxplorer.plot_control.set_axis_by_name("y", y_pixel_param, match_contains=False, block_signals=False)
                if x_set and y_set:
                    image_axes_already_set = True
                # Check for frame parameters in the raw data and setup frame selection
                try:
                    param_names = list(ndxplorer.data_source.parameter_names)
                    t_pixel_param = next((name for name in param_names if "t pixel" in name.lower()), None)
                    z_pixel_param = next((name for name in param_names if "z pixel" in name.lower()), None)
                    
                    frame_param = t_pixel_param or z_pixel_param
                    if frame_param:
                        frame_values = ndxplorer.data_source.values[param_names.index(frame_param), :]
                        n_frames = int(np.max(frame_values)) + 1
                        logging.info("Frame stack detected in image data (%s): %d frames", frame_param, n_frames)
                        ndxplorer.plot_control.setup_frame_selection(frame_param, n_frames)
                    else:
                        ndxplorer.plot_control.hide_frame_selection()
                        
                    # Setup weight parameter for image data
                    photon_param = next((name for name in param_names if "number of photons" in name.lower() or "Number of Photons" in name), None)
                    logging.info("Weight parameter for image data: %s", photon_param)
                    if photon_param:
                        weight_success = ndxplorer.plot_control.set_axis_by_name(
                            "weight", photon_param, match_contains=True, block_signals=True
                        )
                        if weight_success:
                            logging.info("Set weighting to %s for image data", photon_param)
                            try:
                                ndxplorer.weight_param = photon_param
                                ndxplorer.weight_enabled = True
                                # Automatically check the weight checkbox when weight parameter is detected
                                ndxplorer.checkBoxWeight.setChecked(True)
                                logging.info("Automatically enabled weight checkbox for %s", photon_param)
                            except Exception as exc:
                                logging.debug("Failed to enable weight parameter for image data: %s", exc)
                        else:
                            logging.debug("Failed to set weight axis for image data")
                    else:
                        logging.debug("No number of photons parameter found in image data")
                        
                except Exception as e:
                    logging.error(f"Error detecting frames/weights in image data: {e}")
                    ndxplorer.plot_control.hide_frame_selection()
            
            # Disable NaN/Inf masking for image data since pixel coordinates are always valid
            try:
                if hasattr(ndxplorer, 'data_manager') and hasattr(ndxplorer.data_manager, 'selection'):
                    ndxplorer.data_manager.selection.mask_nan = False
                    ndxplorer.data_manager.selection.mask_inf = False
                    logging.info("Disabled NaN/Inf masking for image data")
            except Exception as e:
                logging.debug(f"Could not disable NaN/Inf masking: {e}")
            
            # Clear the stored dims
            delattr(ndxplorer, '_detected_image_dims')
        
        # Log data source info for debugging
        try:
            ds = ndxplorer.data_source
            if ds and not ds.empty:
                param_names = list(ds.parameter_names)
                logging.info(f"Data source has {len(param_names)} parameters: {param_names}")
            else:
                logging.warning("Data source is empty or None during axis detection")
        except Exception as e:
            logging.error(f"Error accessing data source during axis detection: {e}")
        
        # Skip redundant image detection if we already set the axes from raw data
        if not image_axes_already_set:
            # Cancel any in-progress background computations with wrong bins
            if hasattr(ndxplorer.plot_control, '_histogram_worker') and ndxplorer.plot_control._histogram_worker:
                logging.info("Canceling any in-progress histogram computations")
                ndxplorer.plot_control._histogram_worker.cancel()
            
            # Block new histogram requests during axis detection
            old_pending = getattr(ndxplorer.plot_control, '_background_computation_pending', False)
            ndxplorer.plot_control._background_computation_pending = True
            
            try:
                img_applied = ndxplorer.check_and_set_image_axes()
                logging.info(f"Image axis detection result: {img_applied}")
                if not img_applied:
                    try:
                        ndxplorer.apply_default_axes_from_settings()
                    except Exception as exc:  # pragma: no cover - defensive
                        logging.debug("Could not apply default axes: %s", exc)
                
                # Invalidate cache to ensure fresh computation with new bins
                ndxplorer.invalidate_values_cache()
            finally:
                # Restore pending flag
                ndxplorer.plot_control._background_computation_pending = old_pending
        else:
            logging.info("Skipping redundant image detection - axes already set from raw data")
            # Still invalidate cache
            ndxplorer.invalidate_values_cache()
        
        # Clear loading flag IMMEDIATELY to allow histogram computation
        # This must happen before any histogram trigger to prevent blocking
        if hasattr(ndxplorer, 'plot_control'):
            ndxplorer.plot_control._loading_data = False
            logging.info("Cleared _loading_data flag - histogram computation now enabled")
        
        # Trigger histogram update after file loading completes
        # Ensure deferred initialization is complete first
        try:
            # Make sure deferred init is done before updating histograms
            if not getattr(ndxplorer, "_deferred_init_done", False):
                if hasattr(ndxplorer, '_deferred_init'):
                    ndxplorer._deferred_init()
                    logging.info("Triggered deferred init before histogram update")
            
            from ..plotting.plot_update_helpers import update_histograms
            update_histograms(ndxplorer)
            logging.info("Triggered automatic histogram update after file loading")
        except Exception as e:
            logging.warning(f"Failed to trigger automatic histogram update: {e}")
    
    # Use short delay to ensure comboboxes are ready (they're now populated directly)
    QTimer.singleShot(10, _deferred_axis_detection)


def show_merge_dialog(
    ndxplorer: "NDXplorer", title: str
) -> Optional[tuple[bool, str]]:
    """Modal dialog prompting the user for merge behavior."""
    logging.debug("show_merge_dialog")
    dialog = QtWidgets.QDialog(ndxplorer)
    dialog.setWindowTitle(title)
    layout = QtWidgets.QVBoxLayout()

    label = QtWidgets.QLabel("How do you want to merge the new data?")
    layout.addWidget(label)

    replace_rb = QtWidgets.QRadioButton("Replace existing data")
    append_columns_rb = QtWidgets.QRadioButton(
        "Append as columns (add new columns, rows must match)"
    )
    append_rows_rb = QtWidgets.QRadioButton(
        "Append as rows (add new rows of existing columns)"
    )
    replace_rb.setChecked(True)

    layout.addWidget(replace_rb)
    layout.addWidget(append_columns_rb)
    layout.addWidget(append_rows_rb)

    buttons = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
    )
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)

    dialog.setLayout(layout)
    result = dialog.exec_()
    if result != QtWidgets.QDialog.Accepted:
        return None

    append = append_columns_rb.isChecked() or append_rows_rb.isChecked()
    merge_mode = "columns"
    if append_columns_rb.isChecked():
        merge_mode = "columns"
    elif append_rows_rb.isChecked():
        merge_mode = "rows"
    return append, merge_mode
