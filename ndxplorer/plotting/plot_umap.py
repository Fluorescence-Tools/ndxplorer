"""
UMAP plotting functionality for NDXplorer.

This module provides functions for creating and displaying UMAP plots
in separate windows using PyQtGraph.
"""

from typing import Dict, List, Optional, Tuple, Set, Any
import numpy as np
import pandas as pd
import logging
import sys
import io

# Delay import of heavy libraries
umap = None

try:
    from chisurf.gui import QtGui, QtCore, QtWidgets
except ImportError:
    from qtpy import QtCore
    from qtpy import QtGui, QtWidgets

import matplotlib.pyplot as plt

try:
    import pyqtgraph as pg
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False
    pg = None

# Import for 3D plotting
try:
    import pyqtgraph.opengl as gl
    PYQTGRAPH_OPENGL_AVAILABLE = True
except ImportError:
    PYQTGRAPH_OPENGL_AVAILABLE = False
    gl = None
    gl = None

from ..utils.lazy_imports import get_umap


def create_umap_plot(parent, columns: Set[str], params: Dict[str, Any], 
                     data_source, x_values, y_values, z_values, 
                     cluster_labels=None):
    """
    Create and display a UMAP plot in a separate window using PyQtGraph.

    Args:
        parent: The parent widget (used for message boxes and window ownership)
        columns: Set of column names to use for UMAP
        params: Dictionary of parameters for UMAP
            n_neighbors: Number of neighbors to consider for each point
            min_dist: Minimum distance between points in the embedding
            n_components: Number of components (dimensions) for the embedding
        data_source: The data source containing the data to plot
        x_values: The x values for the plot
        y_values: The y values for the plot
        z_values: The z values for the plot
        cluster_labels: Optional array of cluster labels for coloring points
    """
    if not PYQTGRAPH_AVAILABLE or pg is None:
        logging.warning("pyqtgraph not available, cannot create UMAP plot")
        QtWidgets.QMessageBox.warning(
            parent, 
            "UMAP Plot Unavailable", 
            "PyQtGraph is required for UMAP plotting. Please install PyQtGraph to use this feature."
        )
        return None
        
    logging.log(0, f"Creating UMAP plot with params: {params}")

    # Lazy import of umap via centralized getter; offer installation if missing
    from .lazy_imports import get_umap
    if get_umap() is None:
        try:
            from .deps_installer import ensure_package_gui
            desc = (
                "UMAP (Uniform Manifold Approximation and Projection) is a dimensionality "
                "reduction technique used to project high-dimensional data into 2D/3D for visualization."
            )
            installed = ensure_package_gui(
                parent=parent,
                package='umap-learn',
                import_name='umap',
                description=desc,
                allow_pip=True,
                channels=['conda-forge', 'defaults']
            )
        except Exception as _e:
            installed = False
            logging.warning(f"Could not run installer for umap-learn: {_e}")
        if not installed:
            return
        # Retry import after installation
        if get_umap() is None:
            QtWidgets.QMessageBox.information(
                parent,
                "UMAP Installed",
                "UMAP (umap-learn) was installed but could not be imported immediately.\n"
                "Please restart ChiSurf and try again."
            )
            return

    # Store the UMAP windows as instance variables to prevent garbage collection
    if not hasattr(parent, 'umap_windows'):
        parent.umap_windows = []

    # Close any existing UMAP windows
    for window in parent.umap_windows:
        window.close()
    parent.umap_windows = []

    # Get the data for UMAP based on selected columns
    if not columns:
        QtWidgets.QMessageBox.warning(
            parent,
            "Select Columns for UMAP",
            "Please select at least one column before creating a UMAP plot."
        )
        return

    # Use selected columns
    df = data_source.data
    selected_data = []

    for column in columns:
        if column in df.columns:
            # Convert to numeric and handle errors
            values = pd.to_numeric(df[column], errors='coerce').values
            selected_data.append(values)

    if not selected_data:  # If no valid columns were found
        QtWidgets.QMessageBox.warning(
            parent,
            "Invalid Columns for UMAP",
            "The selected columns are not valid or contain no numeric data. Please choose different columns."
        )
        return

    data = np.column_stack(selected_data)

    # Remove any rows with NaN or Inf values
    mask = ~np.any(np.isnan(data) | np.isinf(data), axis=1)
    clean_data = data[mask]

    # Check if we have enough data points
    if len(clean_data) < params['n_neighbors']:
        QtWidgets.QMessageBox.warning(
            parent,
            "Not Enough Data",
            f"Not enough data points for UMAP. Need at least {params['n_neighbors']} (n_neighbors parameter)."
        )
        return

    from .umap_progress import UMAPProgressDialog
    
    # Create and show progress dialog
    progress_dialog = UMAPProgressDialog(parent, "UMAP Plot Computation")
    progress_dialog.show()
    
    # Prepare UMAP parameters
    n_jobs = params.get('n_jobs', -1)  # Default to -1 (all cores) for better performance
    
    umap_params = {
        'n_neighbors': params['n_neighbors'],
        'min_dist': params['min_dist'],
        'n_components': params['n_components'],
        'n_jobs': n_jobs,
        'verbose': True,  # Enable verbose output for progress
        'tqdm_kwds': {'desc': 'UMAP Plot', 'unit': 'epoch'}  # Configure tqdm progress bar
    }
    
    # Add additional parameters from the params dict
    for param_name in ['metric', 'learning_rate', 'init', 'spread', 'low_memory', 
                      'set_op_mix_ratio', 'local_connectivity', 'repulsion_strength',
                      'negative_sample_rate', 'n_epochs']:
        if param_name in params:
            umap_params[param_name] = params[param_name]
    
    # Only set random_state for reproducibility when using single-threaded execution
    # Setting random_state with n_jobs > 1 causes UMAP to override n_jobs to 1
    if n_jobs == 1:
        umap_params['random_state'] = 42
        
    logging.info(f"Creating UMAP reducer for plot with parameters: {umap_params}")
    
    # Start UMAP computation in worker thread
    progress_dialog.run_umap_computation(clean_data, umap_params)
    
    # Show dialog and wait for completion
    result = progress_dialog.exec_()
    
    # Check if computation was successful
    embedding = progress_dialog.get_result()
    error_message = progress_dialog.get_error()
    
    if error_message:
        logging.error(f"Error during UMAP: {error_message}")
        return
        
    if embedding is None:
        logging.error("UMAP computation was cancelled or failed")
        return

    # Create the plot based on the number of components
    if params['n_components'] == 2:
        create_2d_umap_plot(parent, embedding, mask, cluster_labels)
    elif params['n_components'] == 3:
        create_3d_umap_plot(parent, embedding, mask, cluster_labels)


def create_2d_umap_plot(parent, embedding, mask, cluster_labels=None):
    """
    Create and display a 2D UMAP plot.

    Args:
        parent: The parent widget
        embedding: The UMAP embedding (2D array)
        mask: Mask for filtering data points
        cluster_labels: Optional array of cluster labels for coloring points
    """
    # Create a new window for the UMAP plot
    umap_window = QtWidgets.QMainWindow()
    umap_window.setWindowTitle('UMAP Projection')
    umap_window.resize(800, 600)

    # Add the window to the list of UMAP windows
    parent.umap_windows.append(umap_window)

    # Create central widget and layout
    central_widget = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(central_widget)

    # Create plot widget
    plot_widget = pg.PlotWidget(title='UMAP Projection')
    plot_widget.setLabel('bottom', 'UMAP 1')
    plot_widget.setLabel('left', 'UMAP 2')

    # If cluster labels are available, color points by cluster
    if cluster_labels is not None:
        # Get cluster labels for non-filtered points
        filtered_cluster_labels = cluster_labels[mask]

        # Get unique cluster labels
        unique_labels = np.unique(filtered_cluster_labels)

        # Create a colormap
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

        # Create a legend
        legend = pg.LegendItem(offset=(70, 30))
        legend.setParentItem(plot_widget.graphicsItem())

        # Create a scatter plot item for each cluster
        for i, label in enumerate(unique_labels):
            mask_label = filtered_cluster_labels == label

            # Convert color to RGBA format for PyQtGraph
            color = colors[i]
            rgba = (int(color[0]*255), int(color[1]*255), int(color[2]*255), int(color[3]*100))

            scatter_item = pg.ScatterPlotItem(
                x=embedding[mask_label, 0],
                y=embedding[mask_label, 1],
                size=5,
                pen=None,
                brush=pg.mkBrush(*rgba),
                name=f"Cluster {label}"
            )
            plot_widget.addItem(scatter_item)

            # Add item to legend
            legend.addItem(scatter_item, f"Cluster {label}")
    else:
        # Create scatter plot item with default color
        scatter = pg.ScatterPlotItem(
            x=embedding[:, 0],
            y=embedding[:, 1],
            size=5,
            pen=None,
            brush=pg.mkBrush(255, 255, 255, 100)
        )
        plot_widget.addItem(scatter)

    # Add plot widget to layout
    layout.addWidget(plot_widget)

    # Set central widget
    umap_window.setCentralWidget(central_widget)

    # Show the main plot window
    umap_window.show()
    # Bring the window to the front
    umap_window.activateWindow()
    umap_window.raise_()


def create_3d_umap_plot(parent, embedding, mask, cluster_labels=None):
    """
    Create and display a 3D UMAP plot.

    Args:
        parent: The parent widget
        embedding: The UMAP embedding (3D array)
        mask: Mask for filtering data points
        cluster_labels: Optional array of cluster labels for coloring points
    """
    # Check if OpenGL is available
    if gl is None:
        QtWidgets.QMessageBox.warning(
            parent,
            "OpenGL Not Available",
            "PyQtGraph OpenGL is not available. Cannot create 3D plot."
        )
        return

    # Create a new window for the UMAP plot
    umap_window = QtWidgets.QMainWindow()
    umap_window.setWindowTitle('UMAP Projection (3D)')
    umap_window.resize(800, 600)

    # Add the window to the list of UMAP windows
    parent.umap_windows.append(umap_window)

    # Create central widget and layout
    central_widget = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(central_widget)

    # Create 3D view widget
    view_widget = gl.GLViewWidget()

    # If cluster labels are available, color points by cluster
    if cluster_labels is not None:
        # Get cluster labels for non-filtered points
        filtered_cluster_labels = cluster_labels[mask]

        # Get unique cluster labels
        unique_labels = np.unique(filtered_cluster_labels)

        # Create a colormap
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

        # Create a scatter plot for each cluster
        for i, label in enumerate(unique_labels):
            mask_label = filtered_cluster_labels == label

            # Convert color to RGBA format for PyQtGraph
            color = colors[i]

            scatter_item = gl.GLScatterPlotItem(
                pos=embedding[mask_label],
                size=5,
                color=(color[0], color[1], color[2], 0.5),
                pxMode=True
            )
            view_widget.addItem(scatter_item)

        # Create a separate 2D plot widget for the legend
        legend_widget = pg.PlotWidget(title='Legend')
        legend_widget.setFixedHeight(len(unique_labels) * 30 + 50)  # Adjust height based on number of clusters
        legend_widget.getPlotItem().hideAxis('left')
        legend_widget.getPlotItem().hideAxis('bottom')

        # Create a legend
        legend = pg.LegendItem(offset=(10, 10))
        legend.setParentItem(legend_widget.getPlotItem())

        # Add items to the legend
        for i, label in enumerate(unique_labels):
            color = colors[i]
            rgba = (int(color[0]*255), int(color[1]*255), int(color[2]*255), int(color[3]*100))

            # Create a dummy scatter item for the legend
            dummy_scatter = pg.ScatterPlotItem(
                x=[0], y=[0],
                size=5,
                pen=None,
                brush=pg.mkBrush(*rgba)
            )

            # Add to legend
            legend.addItem(dummy_scatter, f"Cluster {label}")

        # Add legend widget to layout
        layout.addWidget(legend_widget)
    else:
        # Create 3D scatter plot with default color
        scatter_plot = gl.GLScatterPlotItem(
            pos=embedding,
            size=5,
            color=(1, 1, 1, 0.5),
            pxMode=True
        )
        view_widget.addItem(scatter_plot)

    # Add axes
    axes = gl.GLAxisItem()
    axes.setSize(x=1, y=1, z=1)
    view_widget.addItem(axes)

    # Add view widget to layout
    layout.addWidget(view_widget)

    # Set central widget
    umap_window.setCentralWidget(central_widget)

    # Show the main plot window
    umap_window.show()
    # Bring the window to the front
    umap_window.activateWindow()
    umap_window.raise_()


__all__ = [
    'create_umap_plot',
    'create_2d_umap_plot',
    'create_3d_umap_plot',
]
