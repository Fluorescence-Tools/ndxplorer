#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Headless CLI interface for ndXplorer filter and image workflows."""

import os
import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import click

# Set up simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def _load_settings_and_equations():
    """Load default equations and constants without Qt GUI dependencies."""
    from ndxplorer.settings import get_settings_path, ensure_default_settings
    
    ensure_default_settings()
    settings_path = get_settings_path()
    
    equations = None
    equations_path = settings_path / "mfd.equations.yaml"
    if equations_path.exists():
        try:
            import yaml
            with open(equations_path, "r", encoding="utf-8") as f:
                equations = yaml.safe_load(f)
        except Exception as e:
            logging.warning(f"Failed to load equations: {e}")
            
    constants = None
    constants_path = settings_path / "mfd.constants.json"
    if constants_path.exists():
        try:
            with open(constants_path, "r", encoding="utf-8") as f:
                constants = json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load constants: {e}")
            
    return constants, equations


def _apply_filter_logic(data_source, select, query):
    """Apply generic selections and query expressions to a DataSource."""
    # Compute columns
    constants, equations = _load_settings_and_equations()
    if constants and equations:
        data_source.compute_columns(constants=constants, equations=equations)
        
    # Build selections
    from ndxplorer.core.data_source import RectangularDataSelection
    selections = []
    for sel_str in select:
        if ":" not in sel_str:
            logging.warning(f"Invalid selection format '{sel_str}'. Expected 'param:min-max'")
            continue
        param, range_str = sel_str.split(":", 1)
        
        # Find column index case-insensitively
        param_idx = None
        for i, col in enumerate(data_source.parameter_names):
            if col.lower() == param.lower() or col.split("|")[0].strip().lower() == param.lower():
                param_idx = i
                break
        if param_idx is None:
            logging.warning(f"Parameter '{param}' not found in data source columns.")
            continue
            
        if "-" not in range_str:
            logging.warning(f"Invalid range format '{range_str}' in '{sel_str}'. Expected 'min-max'")
            continue
        min_val_str, max_val_str = range_str.split("-", 1)
        lower = float(min_val_str) if min_val_str else -np.inf
        upper = float(max_val_str) if max_val_str else np.inf
        
        selections.append(RectangularDataSelection(param_idx, lower, upper))
        
    # Get combined mask
    # True means masked/excluded, False means keep
    mask = data_source.get_mask(selections=selections)
    keep_mask = ~np.any(mask, axis=0)
    
    # Apply pandas query if provided
    if query:
        query_keep = data_source.data.eval(query)
        if isinstance(query_keep, pd.Series):
            query_keep = query_keep.to_numpy()
        keep_mask = keep_mask & query_keep
        
    return keep_mask


@click.group()
def cli():
    """ndXplorer Headless CLI subcommands."""
    pass


@cli.command("filter")
@click.option('--folder', '-d', required=True, type=click.Path(exists=True), help="Burst folder or zip file.")
@click.option('--select', '-s', multiple=True, type=str, help="Selection format: param:min-max")
@click.option('--query', '-q', type=str, help="Pandas eval query string.")
@click.option('--out', '-o', required=True, type=click.Path(), help="Output folder to write filtered bursts.")
@click.option('--skip-nth-row', type=int, default=1, show_default=True, help="Skip every Nth row (1 to load all).")
def filter_cmd(folder, select, query, out, skip_nth_row):
    """Filter bursts by parameters and save the sub-selection."""
    from ndxplorer.io.reader import read_burst_analysis
    from ndxplorer.io.writer import save_burst_ids_headless
    
    logging.info(f"Loading burst analysis from {folder}")
    data_source = read_burst_analysis(folder, skip_nth_row=skip_nth_row)
    
    if data_source.empty:
        logging.error("Loaded data is empty.")
        sys.exit(1)
        
    n_in = data_source.size
    logging.info(f"Loaded {n_in} bursts.")
    
    # Compute keep mask
    keep_mask = _apply_filter_logic(data_source, select, query)
    n_out = int(np.count_nonzero(keep_mask))
    logging.info(f"Filtered to {n_out} bursts.")
    
    # Modify data source in place
    data_source.data = data_source.data.iloc[keep_mask].reset_index(drop=True)
    
    # Save burst IDs headlessly
    save_burst_ids_headless(
        folder_name=out,
        selections=[],
        data_source=data_source
    )
    
    result = {
        "input": str(Path(folder).resolve()),
        "n_in": n_in,
        "n_out": n_out,
        "out": str(Path(out).resolve()),
    }
    click.echo(json.dumps(result, indent=2))


@cli.command("image")
@click.option('--file', '-f', required=True, type=click.Path(exists=True), help="Input MFD-HDF5, CLSM PTU file or folder.")
@click.option('--map', 'map_param', required=True, type=str, help="Parameter map to render (intensity, lifetime, etc.)")
@click.option('--select', '-s', multiple=True, type=str, help="Selection format: param:min-max")
@click.option('--query', '-q', type=str, help="Pandas eval query string.")
@click.option('--roi', type=click.Path(exists=True), help="TIFF file class mask ROI.")
@click.option('--out', '-o', type=click.Path(), help="Rendered map output image path (.png, .tiff).")
@click.option('--out-selection', type=click.Path(), help="Folder to write filtered burst sub-selection from ROI/gate.")
@click.option('--skip-nth-row', type=int, default=1, show_default=True, help="Skip every Nth row (1 to load all).")
def image_cmd(file, map_param, select, query, roi, out, out_selection, skip_nth_row):
    """Load image data, apply selections/ROI, and export maps and selections."""
    from ndxplorer.io.reader import read_mfd_hdf5, read_burst_analysis
    from ndxplorer.core.data_source import MaskDataSelection
    from PIL import Image
    
    # Determine loader by extension
    path = Path(file)
    if path.suffix.lower() in (".h5", ".hdf5"):
        data_source = read_mfd_hdf5([str(path)])
    else:
        # Fallback to burst analysis
        data_source = read_burst_analysis(path, skip_nth_row=skip_nth_row)
        
    if data_source.empty:
        logging.error("Loaded data is empty.")
        sys.exit(1)
        
    n_in = data_source.size
    logging.info(f"Loaded {n_in} points.")
    
    # Setup ROI mask if provided
    roi_selection = None
    n_selected_px = 0
    if roi:
        img = Image.open(roi)
        mask_arr = np.array(img)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[..., 0]
        # Binarize
        mask_arr = (mask_arr > 0).astype(np.int32)
        n_selected_px = int(np.count_nonzero(mask_arr))
        
        # Resolve pixel axes parameters
        param_names = data_source.parameter_names
        x_pixel_param = next((name for name in param_names if "x pixel" in name.lower()), None)
        y_pixel_param = next((name for name in param_names if "y pixel" in name.lower()), None)
        
        if x_pixel_param and y_pixel_param:
            x_idx = param_names.index(x_pixel_param)
            y_idx = param_names.index(y_pixel_param)
            
            x_vals = data_source.values[x_idx, :]
            y_vals = data_source.values[y_idx, :]
            
            x_pixels = int(np.max(x_vals)) + 1
            y_pixels = int(np.max(y_vals)) + 1
            
            edges1 = np.arange(x_pixels + 1)
            edges2 = np.arange(y_pixels + 1)
            
            # The mask needs to be transposed to match Display uses: H.T (ny_bins, nx_bins)
            # If shape is wrong, resize or adapt
            if mask_arr.shape != (y_pixels, x_pixels):
                logging.warning(f"ROI shape {mask_arr.shape} does not match pixel space {(y_pixels, x_pixels)}")
                # Resize image using Pillow
                resized_img = img.resize((x_pixels, y_pixels), resample=Image.NEAREST)
                mask_arr = np.array(resized_img)
                if mask_arr.ndim == 3:
                    mask_arr = mask_arr[..., 0]
                mask_arr = (mask_arr > 0).astype(np.int32)
                
            roi_selection = MaskDataSelection(x_idx, y_idx, mask_arr, edges1, edges2, name="ROI")
            logging.info(f"Loaded ROI mask with shape {mask_arr.shape}")
            
    # Apply filters/queries
    keep_mask = _apply_filter_logic(data_source, select, query)
    
    # If ROI is present, combine it
    if roi_selection is not None:
        roi_mask = roi_selection.get_mask(data_source.values)
        # True means masked out/excluded, so to keep we need not any mask
        keep_mask = keep_mask & (~np.any(roi_mask, axis=0))
        
    n_out = int(np.count_nonzero(keep_mask))
    logging.info(f"Filtered to {n_out} active points.")
    
    # Build parameter map
    param_names = data_source.parameter_names
    x_pixel_param = next((name for name in param_names if "x pixel" in name.lower()), None)
    y_pixel_param = next((name for name in param_names if "y pixel" in name.lower()), None)
    
    if not x_pixel_param or not y_pixel_param:
        logging.error("Image detection failed: X pixel or Y pixel columns not found")
        sys.exit(1)
        
    x_idx = param_names.index(x_pixel_param)
    y_idx = param_names.index(y_pixel_param)
    
    x_vals = data_source.values[x_idx, :]
    y_vals = data_source.values[y_idx, :]
    
    x_pixels = int(np.max(x_vals)) + 1
    y_pixels = int(np.max(y_vals)) + 1
    
    # Filter active coordinates
    active_x = x_vals[keep_mask]
    active_y = y_vals[keep_mask]
    
    if map_param.lower() in ("intensity", "count", "counts"):
        H, _, _ = np.histogram2d(active_x, active_y, bins=[x_pixels, y_pixels], range=[[0, x_pixels], [0, y_pixels]])
        parameter_map = H.T
    else:
        # Find map parameter index
        map_idx = None
        for i, col in enumerate(param_names):
            if col.lower() == map_param.lower() or col.split("|")[0].strip().lower() == map_param.lower():
                map_idx = i
                break
        if map_idx is None:
            logging.error(f"Parameter map '{map_param}' not found in columns")
            sys.exit(1)
            
        param_vals = data_source.values[map_idx, :][keep_mask]
        sum_H, _, _ = np.histogram2d(active_x, active_y, bins=[x_pixels, y_pixels], range=[[0, x_pixels], [0, y_pixels]], weights=param_vals)
        count_H, _, _ = np.histogram2d(active_x, active_y, bins=[x_pixels, y_pixels], range=[[0, x_pixels], [0, y_pixels]])
        
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_H = np.where(count_H > 0, sum_H / count_H, 0.0)
        parameter_map = mean_H.T
        
    # Export parameter map image
    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() in (".tiff", ".tif"):
            # Save quantitative float32 map
            img_out = Image.fromarray(parameter_map.astype(np.float32))
            img_out.save(out_path)
        else:
            # Save normalized visual PNG/JPG
            p_min, p_max = np.min(parameter_map), np.max(parameter_map)
            if p_max > p_min:
                norm_map = (parameter_map - p_min) / (p_max - p_min) * 255.0
            else:
                norm_map = np.zeros_like(parameter_map)
            img_out = Image.fromarray(norm_map.astype(np.uint8))
            img_out.save(out_path)
        logging.info(f"Exported image to {out_path}")
        
    # Export selection if requested
    if out_selection:
        from ndxplorer.io.writer import save_burst_ids_headless
        # Slice original data source data frame
        data_source.data = data_source.data.iloc[keep_mask].reset_index(drop=True)
        save_burst_ids_headless(
            folder_name=out_selection,
            selections=[],
            data_source=data_source
        )
        
    result = {
        "input": str(path.resolve()),
        "map": map_param,
        "shape": [y_pixels, x_pixels],
        "n_selected_px": n_selected_px,
        "out": str(Path(out).resolve()) if out else None,
    }
    click.echo(json.dumps(result, indent=2))
