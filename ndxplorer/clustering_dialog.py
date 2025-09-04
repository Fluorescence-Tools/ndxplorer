"""
Dialog for clustering controls.
Supports keyboard navigation:
- Escape key to close/hide the dialog
"""
from qtpy.QtCore import Signal

# Delay imports of heavy libraries
hdbscan = None
KMeans = None
umap = None

from .logging_config import logging

try:
    from chisurf.gui import QtGui, QtCore, QtWidgets
except ImportError:
    from qtpy import QtCore
    from qtpy import QtGui, QtWidgets

from .column_selection_dialog import ColumnSelectionDialog

_umap_available = True
try:
    import umap as _umap
    logging.info("Imported umap library")
except ImportError:
    _umap_available = False


class ClusteringDialog(QtWidgets.QDialog):
    """
    Dialog for clustering controls.
    """
    # Signal emitted when clustering is done
    clustering_done = Signal(tuple)
    # Signal emitted when an error occurs
    clustering_error = Signal(str)
    # Signal emitted to report progress
    progress_updated = Signal(int)

    def __init__(self, parent=None):
        logging.log(0, "Initializing ClusteringDialog")
        super(ClusteringDialog, self).__init__(parent)
        self.setWindowTitle("Clustering Controls")

        # Initialize clustering settings
        self._cluster_method = "kmeans"  # default to K-means

        # HDBSCAN specific parameters
        self._cluster_min_samples = 5
        self._cluster_min_cluster_size = 50

        # K-means specific parameters
        self._cluster_n_clusters = 3

        # UMAP specific parameters
        self._umap_n_neighbors = 15
        self._umap_min_dist = 0.1
        self._umap_n_components = 2
        self._use_umap_enhancement = False  # default to not using UMAP enhancement

        # Common clustering variables
        self._cluster_columns = set()
        self.clustering_worker = None

        # Create the UI
        self.setup_ui()

    def closeEvent(self, event):
        """
        Override the close event to hide the dialog instead of closing it.
        This prevents the dialog from being deleted when closed.
        """
        logging.log(0, "ClusteringDialog closeEvent - hiding dialog instead of closing")
        event.ignore()
        self.hide()

    def keyPressEvent(self, event):
        """
        Handle keyboard events.
        - Escape key: hide the dialog
        """
        key = event.key()

        if key == QtCore.Qt.Key_Escape:
            logging.log(0, "ClusteringDialog keyPressEvent - Escape key pressed, hiding dialog")
            self.hide()
            event.accept()
        else:
            # Pass other keys to parent class
            super(ClusteringDialog, self).keyPressEvent(event)

    def setup_ui(self):
        """Set up the dialog UI."""
        logging.log(0, "Setting up ClusteringDialog UI")
        # Main layout
        main_layout = QtWidgets.QVBoxLayout(self)

        # Add help text for keyboard navigation
        help_label = QtWidgets.QLabel("Press Escape key to close this dialog")
        help_label.setStyleSheet("color: #666666; font-size: 10pt;")
        main_layout.addWidget(help_label)

        # Create dropdown for selecting clustering method
        method_layout = QtWidgets.QHBoxLayout()
        method_label = QtWidgets.QLabel("Method:")
        self.comboBoxClusteringMethod = QtWidgets.QComboBox()
        self.comboBoxClusteringMethod.addItems(["kmeans", "hdbscan"])
        self.comboBoxClusteringMethod.setCurrentText(self._cluster_method)
        self.comboBoxClusteringMethod.currentTextChanged.connect(self.on_clustering_method_changed)
        method_layout.addWidget(method_label)
        method_layout.addWidget(self.comboBoxClusteringMethod)

        # Create container widgets for different parameter sets
        self.hdbscan_container = QtWidgets.QWidget()
        self.kmeans_container = QtWidgets.QWidget()

        # Create form layouts for each container
        hdbscan_layout = QtWidgets.QFormLayout(self.hdbscan_container)
        kmeans_layout = QtWidgets.QFormLayout(self.kmeans_container)

        # Create widgets for HDBSCAN parameters
        self.spinBoxMinSamples = QtWidgets.QSpinBox()
        self.spinBoxMinSamples.setRange(1, 100)
        self.spinBoxMinSamples.setValue(self._cluster_min_samples)
        self.spinBoxMinSamples.valueChanged.connect(self.on_min_samples_changed)

        self.spinBoxMinClusterSize = QtWidgets.QSpinBox()
        self.spinBoxMinClusterSize.setRange(1, 100)
        self.spinBoxMinClusterSize.setValue(self._cluster_min_cluster_size)
        self.spinBoxMinClusterSize.valueChanged.connect(self.on_min_cluster_size_changed)

        # Add HDBSCAN widgets to its layout
        hdbscan_layout.addRow("Min Samples:", self.spinBoxMinSamples)
        hdbscan_layout.addRow("Min Cluster Size:", self.spinBoxMinClusterSize)

        # Create widgets for K-means parameters
        self.spinBoxNClusters = QtWidgets.QSpinBox()
        self.spinBoxNClusters.setRange(1, 20)
        self.spinBoxNClusters.setValue(self._cluster_n_clusters)
        self.spinBoxNClusters.valueChanged.connect(self.on_n_clusters_changed)

        # Add K-means widgets to its layout
        kmeans_layout.addRow("Number of Clusters:", self.spinBoxNClusters)

        # Create widgets for UMAP parameters
        self.spinBoxUMAPNeighbors = QtWidgets.QSpinBox()
        self.spinBoxUMAPNeighbors.setRange(2, 100)
        self.spinBoxUMAPNeighbors.setValue(self._umap_n_neighbors)
        self.spinBoxUMAPNeighbors.valueChanged.connect(self.on_umap_n_neighbors_changed)

        self.doubleSpinBoxUMAPMinDist = QtWidgets.QDoubleSpinBox()
        self.doubleSpinBoxUMAPMinDist.setRange(0.0, 1.0)
        self.doubleSpinBoxUMAPMinDist.setSingleStep(0.01)
        self.doubleSpinBoxUMAPMinDist.setValue(self._umap_min_dist)
        self.doubleSpinBoxUMAPMinDist.valueChanged.connect(self.on_umap_min_dist_changed)

        self.spinBoxUMAPComponents = QtWidgets.QSpinBox()
        self.spinBoxUMAPComponents.setRange(2, 3)  # Limit to 2D or 3D for visualization
        self.spinBoxUMAPComponents.setValue(self._umap_n_components)
        self.spinBoxUMAPComponents.valueChanged.connect(self.on_umap_n_components_changed)

        # Create a parameters layout to hold the containers
        parameters_layout = QtWidgets.QVBoxLayout()
        parameters_layout.addWidget(self.hdbscan_container)
        parameters_layout.addWidget(self.kmeans_container)

        # Add parameters layout to main layout
        main_layout.addLayout(method_layout)
        main_layout.addLayout(parameters_layout)

        # Initialize visibility based on current method
        self.update_clustering_parameters_ui()

        # Create button to select columns for clustering
        self.pushButtonSelectColumns = QtWidgets.QPushButton()
        self.pushButtonSelectColumns.setText("Select Columns (Recommended)")
        self.pushButtonSelectColumns.setToolTip("It is highly recommended to select specific columns for clustering to get better results.")
        self.pushButtonSelectColumns.setStyleSheet("background-color: #e6f2ff; font-weight: bold;")
        self.pushButtonSelectColumns.clicked.connect(self.on_select_columns)

        # Create button to apply clustering
        self.pushButtonApplyClustering = QtWidgets.QPushButton()
        self.pushButtonApplyClustering.setText("Apply Clustering")
        self.pushButtonApplyClustering.clicked.connect(self.on_apply_clustering)

        # Create cancel button (initially hidden)
        self.pushButtonCancelClustering = QtWidgets.QPushButton()
        self.pushButtonCancelClustering.setText("Cancel Clustering")
        self.pushButtonCancelClustering.clicked.connect(self.on_cancel_clustering)
        self.pushButtonCancelClustering.setVisible(False)

        # Create save button
        self.pushButtonSaveClustering = QtWidgets.QPushButton()
        self.pushButtonSaveClustering.setText("Save Clustering Data")
        self.pushButtonSaveClustering.clicked.connect(self.on_save_clustering_data)
        self.pushButtonSaveClustering.setEnabled(False)  # Initially disabled until clustering is done

        # Create UMAP plot button
        self.pushButtonUMAPPlot = QtWidgets.QPushButton()
        self.pushButtonUMAPPlot.setText("Create UMAP Plot")
        self.pushButtonUMAPPlot.clicked.connect(self.on_create_umap_plot)

        # Create a group box for UMAP settings
        self.groupBoxUMAP = QtWidgets.QGroupBox("UMAP Settings")
        umap_layout = QtWidgets.QFormLayout()

        # Add checkbox for UMAP enhancement
        self.checkBoxUMAPEnhancement = QtWidgets.QCheckBox("Enhance clustering with UMAP")
        self.checkBoxUMAPEnhancement.setToolTip("Use UMAP for dimensionality reduction before applying clustering algorithm")
        self.checkBoxUMAPEnhancement.setChecked(self._use_umap_enhancement)
        self.checkBoxUMAPEnhancement.stateChanged.connect(self.on_umap_enhancement_changed)
        umap_layout.addRow(self.checkBoxUMAPEnhancement)

        umap_layout.addRow("Number of Neighbors:", self.spinBoxUMAPNeighbors)
        umap_layout.addRow("Minimum Distance:", self.doubleSpinBoxUMAPMinDist)
        umap_layout.addRow("Number of Components:", self.spinBoxUMAPComponents)
        self.groupBoxUMAP.setLayout(umap_layout)

        # Create progress bar (initially hidden)
        self.progressBarClustering = QtWidgets.QProgressBar()
        self.progressBarClustering.setRange(0, 100)
        self.progressBarClustering.setValue(0)
        self.progressBarClustering.setVisible(False)

        # Add widgets to main layout
        main_layout.addWidget(self.pushButtonSelectColumns)
        main_layout.addWidget(self.pushButtonApplyClustering)
        main_layout.addWidget(self.pushButtonCancelClustering)
        main_layout.addWidget(self.pushButtonSaveClustering)
        main_layout.addWidget(self.groupBoxUMAP)
        main_layout.addWidget(self.pushButtonUMAPPlot)
        main_layout.addWidget(self.progressBarClustering)

    def update_clustering_parameters_ui(self):
        """
        Update the parameter form based on the selected clustering method.
        """
        logging.log(0, f"Updating clustering parameters UI for method: {self._cluster_method}")

        # Show/hide containers based on the selected method
        if self._cluster_method == "hdbscan":
            self.hdbscan_container.setVisible(True)
            self.kmeans_container.setVisible(False)
        elif self._cluster_method == "kmeans":
            self.hdbscan_container.setVisible(False)
            self.kmeans_container.setVisible(True)


    def on_min_samples_changed(self, value):
        """
        Handle changes to the min_samples parameter.
        """
        logging.log(0, f"Changing min_samples to {value}")
        self._cluster_min_samples = value

    def on_min_cluster_size_changed(self, value):
        """
        Handle changes to the min_cluster_size parameter.
        """
        logging.log(0, f"Changing min_cluster_size to {value}")
        self._cluster_min_cluster_size = value

    def on_n_clusters_changed(self, value):
        """
        Handle changes to the n_clusters parameter.
        """
        logging.log(0, f"Changing n_clusters to {value}")
        self._cluster_n_clusters = value

    def on_umap_n_neighbors_changed(self, value):
        """
        Handle changes to the UMAP n_neighbors parameter.
        """
        logging.log(0, f"Changing UMAP n_neighbors to {value}")
        self._umap_n_neighbors = value

    def on_umap_min_dist_changed(self, value):
        """
        Handle changes to the UMAP min_dist parameter.
        """
        logging.log(0, f"Changing UMAP min_dist to {value}")
        self._umap_min_dist = value

    def on_umap_n_components_changed(self, value):
        """
        Handle changes to the UMAP n_components parameter.
        """
        logging.log(0, f"Changing UMAP n_components to {value}")
        self._umap_n_components = value

    def on_umap_enhancement_changed(self, state):
        """
        Handle changes to the UMAP enhancement checkbox.
        """
        logging.log(0, f"Changing UMAP enhancement to {bool(state)}")
        self._use_umap_enhancement = bool(state)

    def on_clustering_method_changed(self, method):
        """
        Handle changes to the clustering method.
        """
        logging.log(0, f"Changing clustering method to {method}")
        self._cluster_method = method
        self.update_clustering_parameters_ui()

    def on_select_columns(self):
        """
        Open a dialog to select columns for clustering.
        """
        logging.log(0, "Opening column selection dialog for clustering")
        # Get current parameter names from parent
        parent = self.parent()
        if parent is not None and hasattr(self.parent(), 'data_source'):
            parameter_names = parent.data_source.parameter_names
            print("onllslslsls")

            # Create and show the dialog
            dialog = ColumnSelectionDialog(
                parent=self,
                column_names=parameter_names,
                selected_columns=self._cluster_columns
            )

            # If dialog is accepted, update selected columns
            if dialog.exec_():
                self._cluster_columns = dialog.get_selected_columns()

                # Update button text to show number of selected columns
                num_selected = len(self._cluster_columns)
                if num_selected > 0:
                    self.pushButtonSelectColumns.setText(f"Select Columns (Recommended) ({num_selected})")
                else:
                    self.pushButtonSelectColumns.setText("Select Columns (Recommended)")

    def on_apply_clustering(self):
        """
        Apply clustering with current parameters.
        """
        logging.log(0, f"Applying clustering with method: {self._cluster_method}, columns: {self._cluster_columns}")
        # Check if the required library is available
        if self._cluster_method == "hdbscan":
            # Lazy import of hdbscan
            import hdbscan

            if not hdbscan:
                QtWidgets.QMessageBox.warning(
                    QtWidgets.QApplication.activeWindow(),
                    "HDBSCAN Not Available",
                    "HDBSCAN is not installed. Please install it using pip or conda."
                )
                return
        elif self._cluster_method == "kmeans":
            # Lazy import of KMeans
            try:
                from sklearn.cluster import KMeans as _KMeans
                logging.info("Imported KMeans library")
                _kmeans_available = True
            except ImportError:
                _kmeans_available = False

            if not _kmeans_available:
                QtWidgets.QMessageBox.warning(
                QtWidgets.QApplication.activeWindow(),
                "scikit-learn Not Available",
                "scikit-learn is not installed. Please install it using pip or conda."
            )
                return

        # Check if any columns are selected for clustering
        if not self._cluster_columns:
            # No columns selected, ask user if they want to use default (x, y, z) values
            reply = QtWidgets.QMessageBox.question(
                QtWidgets.QApplication.activeWindow(),
                "Select Columns for Clustering",
                "It is recommended to select specific columns for clustering to get better results.\n\n"
                "Would you like to select columns now?\n\n"
                "If you click 'No', clustering will use only the current X, Y, and Z axis values, "
                "which may not provide optimal clustering results.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes
            )

            if reply == QtWidgets.QMessageBox.Yes:
                # Open column selection dialog
                self.on_select_columns()

                # If still no columns selected after dialog, return
                if not self._cluster_columns:
                    return 
            # If user clicked No, continue with clustering using X, Y, Z values

        # Update UI for clustering in progress
        self.pushButtonApplyClustering.setEnabled(False)
        self.pushButtonApplyClustering.setVisible(False)
        self.pushButtonSelectColumns.setEnabled(False)
        self.comboBoxClusteringMethod.setEnabled(False)
        self.pushButtonCancelClustering.setVisible(True)
        self.progressBarClustering.setVisible(True)
        self.progressBarClustering.setValue(0)

        # Notify parent to start clustering
        if self.parent() is not None and hasattr(self.parent(), 'start_clustering_from_dialog'):
            # Prepare parameters based on the selected method
            params = {}
            if self._cluster_method == "hdbscan":
                params = {
                    "min_samples": self._cluster_min_samples,
                    "min_cluster_size": self._cluster_min_cluster_size
                }
            elif self._cluster_method == "kmeans":
                params = {
                    "n_clusters": self._cluster_n_clusters
                }

            # Add UMAP enhancement parameters if enabled
            if self._use_umap_enhancement:
                params["use_umap_enhancement"] = True
                params["umap_n_neighbors"] = self._umap_n_neighbors
                params["umap_min_dist"] = self._umap_min_dist
                params["umap_n_components"] = self._umap_n_components

            self.parent().start_clustering_from_dialog(
                self._cluster_method,
                self._cluster_columns,
                params
            )

    def on_cancel_clustering(self):
        """
        Cancel the current clustering operation.
        """
        logging.log(0, "Cancelling clustering operation")
        if self.parent() is not None and hasattr(self.parent(), 'cancel_clustering'):
            self.parent().cancel_clustering()

        # Update UI
        self.pushButtonCancelClustering.setText("Cancelling...")
        self.pushButtonCancelClustering.setEnabled(False)

    def on_save_clustering_data(self):
        """
        Save clustering data to a folder.
        """
        logging.log(0, "Saving clustering data to file")
        if self.parent() is not None and hasattr(self.parent(), 'onSaveClusteringData'):
            self.parent().onSaveClusteringData()

    def on_create_umap_plot(self):
        """
        Create and display a UMAP plot in a separate window.
        """
        logging.log(0, "Creating UMAP plot")

        # Check if UMAP is available
        if not _umap_available:
            QtWidgets.QMessageBox.warning(
                QtWidgets.QApplication.activeWindow(),
                "UMAP Not Available",
                "UMAP is not installed. Please install it using pip or conda."
            )
            return

        do_umap = True
        # Require at least two columns to be selected for UMAP/clustering
        if not self._cluster_columns or len(self._cluster_columns) < 2:
            do_umap = False
            QtWidgets.QMessageBox.warning(
                QtWidgets.QApplication.activeWindow(),
                "Select Columns for UMAP",
                "Please select at least two columns before creating a UMAP plot."
            )
            logging.log(0, "At least two columns are required for UMAP/clustering")

        # Notify parent to create UMAP plot
        if self.parent() is not None and hasattr(self.parent(), 'create_umap_plot') and do_umap:
            # Prepare parameters for UMAP
            params = {
                "n_neighbors": self._umap_n_neighbors,
                "min_dist": self._umap_min_dist,
                "n_components": self._umap_n_components
            }

            self.parent().create_umap_plot(
                self._cluster_columns,
                params
            )

    def update_progress(self, progress):
        """
        Update the progress bar with the current clustering progress.
        """
        logging.log(0, f"Updating clustering progress: {progress}%")
        self.progressBarClustering.setValue(progress)

    def clustering_completed(self, success=True):
        """
        Update UI when clustering is completed.
        """
        logging.log(0, f"Clustering completed with success={success}")
        # Reset UI
        self.pushButtonApplyClustering.setEnabled(True)
        self.pushButtonApplyClustering.setVisible(True)
        self.pushButtonSelectColumns.setEnabled(True)
        self.comboBoxClusteringMethod.setEnabled(True)
        self.pushButtonCancelClustering.setVisible(False)
        self.pushButtonCancelClustering.setText("Cancel Clustering")
        self.pushButtonCancelClustering.setEnabled(True)
        self.progressBarClustering.setVisible(False)
        self.progressBarClustering.setValue(0)
        self.pushButtonSaveClustering.setEnabled(success)
