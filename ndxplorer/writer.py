from __future__ import print_function
from typing import List, Dict, Optional, Set
from pathlib import Path
try:
    from chisurf import logging
except ImportError:
    import logging
import numpy as np
import json
import pandas as pd
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QProgressBar, QLabel, QApplication
from PyQt5.QtCore import Qt, QCoreApplication
from .data_source import DataSource, DataSelection


class ProgressWindow(QDialog):
    def __init__(self, title="Progress", message="Processing...", max_value=100, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModal)
        self.layout = QVBoxLayout()
        self.label = QLabel(message)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, max_value)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.progress_bar)
        self.setLayout(self.layout)

    def set_value(self, value: int):
        self.progress_bar.setValue(value)


def save_burst_ids(
        folder_name: str,
        selections: List[DataSelection],
        data_source: DataSource
):
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    folder_path = Path(folder_name)
    folder_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving burst IDs to {folder_path}")

    df = data_source.data
    mask = data_source.get_mask(selections=selections)

    mask_flat = np.sum(mask, axis=0).astype(bool)
    mas = np.broadcast_to(mask_flat, (df.shape[1], df.shape[0]))
    dm = df.mask(mas.T)

    dm = dm.loc[dm['First File'] == dm['Last File']]
    grouped = dm.groupby('First File')

    total_files = len(grouped)
    progress_window = ProgressWindow(title="Saving Files", message="Saving Burst ID files...", max_value=total_files)
    progress_window.show()

    for i, (filename, g) in enumerate(grouped, start=1):
        fn = Path(filename).name
        ext = Path(filename).suffix
        bst_file = folder_path / f"{fn}.bst"

        a = np.vstack([g["First Photon"], g["Last Photon"]]).astype(int)
        np.savetxt(bst_file, a.T, fmt='%i', delimiter='\t')

        logging.info(f"Saved burst ID file: {bst_file}")
        progress_window.set_value(i)
        QCoreApplication.processEvents()

    progress_window.set_value(total_files)
    progress_window.close()


def save_clustering_data(
        folder_name: str,
        data_source: DataSource,
        cluster_method: str,
        cluster_labels: Optional[np.ndarray],
        cluster_probabilities: Optional[np.ndarray],
        cluster_columns: Set[str],
        parameters: Dict
):
    """
    Save clustering data to a folder.

    Args:
        folder_name: Path to the folder where clustering data will be saved
        data_source: DataSource object containing the data
        cluster_method: Clustering method used (e.g., 'kmeans', 'hdbscan')
        cluster_labels: Array of cluster labels for each data point
        cluster_probabilities: Array of cluster membership probabilities
        cluster_columns: Set of column names used for clustering
        parameters: Dictionary of clustering parameters
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    # Create the clustering folder
    folder_path = Path(folder_name)
    clustering_folder = folder_path / "clustering"
    clustering_folder.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving clustering data to {clustering_folder}")

    # Create a progress window
    progress_window = ProgressWindow(title="Saving Clustering Data", message="Saving clustering data...", max_value=3)
    progress_window.show()
    QCoreApplication.processEvents()

    try:
        # Step 1: Save clustering parameters
        progress_window.set_value(1)
        progress_window.label.setText("Saving clustering parameters...")
        QCoreApplication.processEvents()

        # Create a dictionary with all clustering information
        clustering_info = {
            "method": cluster_method,
            "columns": list(cluster_columns),
            "parameters": parameters,
            "timestamp": pd.Timestamp.now().isoformat()
        }

        # Save parameters to a JSON file
        params_file = clustering_folder / "clustering_parameters.json"
        with open(params_file, 'w') as f:
            json.dump(clustering_info, f, indent=4)

        logging.info(f"Saved clustering parameters to {params_file}")

        # Step 2: Save cluster labels
        progress_window.set_value(2)
        progress_window.label.setText("Saving cluster labels...")
        QCoreApplication.processEvents()

        if cluster_labels is not None:
            # Get the data frame
            df = data_source.data.copy()

            # Add cluster labels and probabilities if they don't exist
            if 'Cluster Label' not in df.columns:
                df['Cluster Label'] = cluster_labels

            if cluster_probabilities is not None and 'Cluster Probability' not in df.columns:
                df['Cluster Probability'] = cluster_probabilities

            # Save the full data with cluster labels to a CSV file
            full_data_file = clustering_folder / "clustering_full_data.csv"
            df.to_csv(full_data_file, index=False)
            logging.info(f"Saved full data with cluster labels to {full_data_file}")

            # Save just the cluster information (ID, label, probability)
            cluster_info = pd.DataFrame({
                'ID': range(len(cluster_labels)),
                'Cluster Label': cluster_labels
            })

            if cluster_probabilities is not None:
                cluster_info['Cluster Probability'] = cluster_probabilities

            cluster_info_file = clustering_folder / "cluster_labels.csv"
            cluster_info.to_csv(cluster_info_file, index=False)
            logging.info(f"Saved cluster labels to {cluster_info_file}")

            # Save data for each cluster separately
            unique_labels = np.unique(cluster_labels)
            for label in unique_labels:
                if label >= 0:  # Skip noise points (label -1)
                    cluster_mask = cluster_labels == label
                    cluster_df = df[cluster_mask]
                    cluster_file = clustering_folder / f"cluster_{label}.csv"
                    cluster_df.to_csv(cluster_file, index=False)
                    logging.info(f"Saved data for cluster {label} to {cluster_file}")
        else:
            logging.warning("No cluster labels to save")

        # Step 3: Complete
        progress_window.set_value(3)
        progress_window.label.setText("Clustering data saved successfully!")
        QCoreApplication.processEvents()

        logging.info("Clustering data saved successfully")

    except Exception as e:
        logging.error(f"Error saving clustering data: {str(e)}")
    finally:
        progress_window.close()
