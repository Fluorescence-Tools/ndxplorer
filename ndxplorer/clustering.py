from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import logging

# Delay imports of heavy libraries
hdbscan = None
KMeans = None
umap = None

from qtpy.QtCore import QThread, Signal


class ClusteringWorker(QThread):
    """Worker class for performing clustering in a separate thread"""
    # Signal emitted when clustering is done
    clustering_done = Signal(tuple)
    # Signal emitted when an error occurs
    clustering_error = Signal(str)
    # Signal emitted to report progress
    progress_updated = Signal(int)

    def __init__(self, parent, method, params):
        super().__init__(parent)
        self.parent = parent
        self.method = method
        self.params = params
        self._stop_requested = False

    def stop(self):
        """Request the worker to stop processing"""
        self._stop_requested = True
        logging.info("Clustering stop requested")

    def run(self):
        try:
            # Perform clustering in the worker thread
            result = self.parent.perform_clustering(
                method=self.method,
                **self.params,
                worker=self  # Pass the worker instance to allow progress updates and cancellation
            )
            # Emit signal with the result only if not stopped
            if not self._stop_requested:
                self.clustering_done.emit(result)
        except Exception as e:
            # Log the error
            logging.error(f"Error in clustering worker thread: {str(e)}")
            # Emit error signal only if not stopped
            if not self._stop_requested:
                self.clustering_error.emit(str(e))
                # Emit done signal with None values to ensure the UI is updated
                self.clustering_done.emit((None, None))


class ClusteringManager:
    """Class to manage clustering operations"""

    def __init__(self, data_source=None):
        """
        Initialize the clustering manager.

        Args:
            data_source: The data source to use for clustering
        """
        self._data_source = data_source
        self._cluster_data_shape = None
        self._clustering_worker = None

    def perform_clustering(self, method=None, worker=None, **kwargs) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Perform clustering on the current data using the specified method.

        Args:
            method: The clustering method to use. If None, uses the method from clustering dialog.
            worker: The ClusteringWorker instance that called this method, used for progress updates and cancellation.
            **kwargs: Additional parameters for the clustering method.
                For HDBSCAN:
                    min_samples: Minimum number of samples in a neighborhood for a point to be considered a core point.
                    min_cluster_size: Minimum number of points for a cluster.
                For K-means:
                    n_clusters: Number of clusters to form.
                For UMAP enhancement:
                    use_umap_enhancement: Whether to use UMAP for dimensionality reduction before clustering.
                    umap_n_neighbors: Number of neighbors to consider for each point in UMAP.
                    umap_min_dist: Minimum distance between points in the UMAP embedding.
                    umap_n_components: Number of components (dimensions) for the UMAP embedding.
                For data selection:
                    data: The data to cluster (numpy array)
                    columns: The column names for the data
                    mask: Mask for filtering data points

        Returns:
            Tuple containing:
            - cluster_labels: Array of cluster labels for each data point
            - cluster_probabilities: Array of cluster membership probabilities (or None for K-means)
        """
        if method is None:
            method = kwargs.get("method", "hdbscan")

        # Library availability will be checked lazily below for each method

        # Get data from kwargs or data_source
        data = kwargs.get("data", None)
        if data is None and self._data_source is not None and not self._data_source.empty:
            # Get the data for clustering based on selected columns
            columns = kwargs.get("columns", [])
            if columns:
                # Use selected columns
                df = self._data_source.data
                selected_data = []

                for column in columns:
                    if column in df.columns:
                        # Convert to numeric and handle errors
                        values = pd.to_numeric(df[column], errors='coerce').values
                        selected_data.append(values)

                if selected_data:
                    data = np.column_stack(selected_data)

            # If no data was collected, use x, y, z values if provided
            if data is None:
                x_values = kwargs.get("x_values", None)
                y_values = kwargs.get("y_values", None)
                z_values = kwargs.get("z_values", None)

                if x_values is not None and y_values is not None and z_values is not None:
                    data = np.column_stack((x_values, y_values, z_values))

        if data is None or len(data) == 0:
            logging.error("No data available for clustering.")
            return None, None

        # Report progress: 10% - Starting data preparation
        if worker:
            worker.progress_updated.emit(10)
            # Check if stop was requested
            if worker._stop_requested:
                logging.info("Clustering cancelled during data preparation")
                return None, None

        # Report progress: 20% - Data collected
        if worker:
            worker.progress_updated.emit(20)
            # Check if stop was requested
            if worker._stop_requested:
                logging.info("Clustering cancelled after data collection")
                return None, None

        # Remove any rows with NaN or Inf values
        mask = ~np.any(np.isnan(data) | np.isinf(data), axis=1)
        clean_data = data[mask]

        # Report progress: 30% - Data cleaned
        if worker:
            worker.progress_updated.emit(30)
            # Check if stop was requested
            if worker._stop_requested:
                logging.info("Clustering cancelled after data cleaning")
                return None, None

        # Check if UMAP enhancement is enabled
        use_umap_enhancement = kwargs.get("use_umap_enhancement", False)

        # Apply UMAP dimensionality reduction if enhancement is enabled
        if use_umap_enhancement:
            # Lazy import of umap
            try:
                import umap as _umap
                logging.info("Imported umap library")
                _umap_available = True
            except ImportError:
                logging.error("UMAP is not installed. Cannot perform dimensionality reduction.")
                _umap_available = False

            if _umap_available:
                logging.info("Applying UMAP dimensionality reduction before clustering")

                # Get UMAP parameters
                umap_n_neighbors = kwargs.get("umap_n_neighbors", 15)
                umap_min_dist = kwargs.get("umap_min_dist", 0.1)
                umap_n_components = kwargs.get("umap_n_components", 2)

                # Check if we have enough data points for UMAP
                if len(clean_data) < umap_n_neighbors:
                    logging.warning(f"Not enough data points for UMAP. Need at least {umap_n_neighbors} (n_neighbors parameter).")
                    return None, None

                # Report progress: 35% - Starting UMAP
                if worker:
                    worker.progress_updated.emit(35)
                    # Check if stop was requested
                    if worker._stop_requested:
                        logging.info("Clustering cancelled before UMAP")
                        return None, None

                try:
                    # Create and fit the UMAP reducer
                    reducer = _umap.UMAP(
                        n_neighbors=umap_n_neighbors,
                        min_dist=umap_min_dist,
                        n_components=umap_n_components,
                        random_state=42  # For reproducibility
                    )

                    # Fit and transform the data
                    clean_data = reducer.fit_transform(clean_data)
                    logging.info(f"Data dimensionality reduced to {umap_n_components} using UMAP")

                except Exception as e:
                    logging.error(f"Error during UMAP dimensionality reduction: {str(e)}")
                    # Continue with original data if UMAP fails
                    logging.info("Continuing with original data")
            else:
                return None, None

        # Extract parameters for the selected method
        if method == "hdbscan":
            min_samples = kwargs.get("min_samples", 5)
            min_cluster_size = kwargs.get("min_cluster_size", 5)

            # Check if we have enough data points
            if len(clean_data) < min_cluster_size:
                logging.warning(f"Not enough data points for HDBSCAN clustering. Need at least {min_cluster_size}.")
                return None, None
        elif method == "kmeans":
            n_clusters = kwargs.get("n_clusters", 2)

            # Check if we have enough data points
            if len(clean_data) < n_clusters:
                logging.warning(f"Not enough data points for K-means clustering. Need at least {n_clusters} (one per cluster).")
                return None, None

        try:
            # Report progress: 40% - Starting clustering algorithm
            if worker:
                worker.progress_updated.emit(40)
                # Check if stop was requested
                if worker._stop_requested:
                    logging.info("Clustering cancelled before algorithm start")
                    return None, None

            if method == "hdbscan":
                # Lazy import of hdbscan
                try:
                    import hdbscan as _hdbscan
                    logging.info("Imported hdbscan library")
                except ImportError:
                    logging.error("HDBSCAN is not installed. Cannot perform clustering.")
                    return None, None

                # Create and fit the HDBSCAN clusterer
                clusterer = _hdbscan.HDBSCAN(
                    min_samples=min_samples,
                    min_cluster_size=min_cluster_size,
                    prediction_data=True
                )

                # This is the most CPU-intensive part
                clusterer.fit(clean_data)

                # Report progress: 70% - HDBSCAN clustering completed
                if worker:
                    worker.progress_updated.emit(70)
                    # Check if stop was requested
                    if worker._stop_requested:
                        logging.info("Clustering cancelled after HDBSCAN fit")
                        return None, None

                # Get cluster labels and probabilities
                labels = clusterer.labels_
                probabilities = clusterer.probabilities_

                # Create full-sized arrays with NaN for filtered points
                full_labels = np.full(len(data), -1, dtype=np.int32)
                full_probabilities = np.zeros(len(data))

                # Fill in the values for non-filtered points
                full_labels[mask] = labels
                full_probabilities[mask] = probabilities

            elif method == "kmeans":
                # Lazy import of KMeans
                try:
                    from sklearn.cluster import KMeans as _KMeans
                    logging.info("Imported KMeans library")
                except ImportError:
                    logging.error("scikit-learn is not installed. Cannot perform clustering.")
                    return None, None

                # Create and fit the K-means clusterer
                clusterer = _KMeans(
                    n_clusters=n_clusters,
                    random_state=42  # For reproducibility
                )

                # This is the most CPU-intensive part
                clusterer.fit(clean_data)

                # Report progress: 70% - K-means clustering completed
                if worker:
                    worker.progress_updated.emit(70)
                    # Check if stop was requested
                    if worker._stop_requested:
                        logging.info("Clustering cancelled after K-means fit")
                        return None, None

                # Get cluster labels
                labels = clusterer.labels_

                # Create full-sized arrays with NaN for filtered points
                full_labels = np.full(len(data), -1, dtype=np.int32)

                # Fill in the values for non-filtered points
                full_labels[mask] = labels

                # For K-means, we don't have probabilities, so we use the distance to the cluster center
                # as a proxy for probability (inverse of distance)
                distances = np.zeros(len(clean_data))

                # Report progress: 80% - Starting distance calculations
                if worker:
                    worker.progress_updated.emit(80)
                    # Check if stop was requested
                    if worker._stop_requested:
                        logging.info("Clustering cancelled before distance calculations")
                        return None, None

                # Calculate distances in batches to reduce CPU load and allow cancellation
                batch_size = 1000
                for batch_start in range(0, len(clean_data), batch_size):
                    batch_end = min(batch_start + batch_size, len(clean_data))

                    # Check if stop was requested before processing each batch
                    if worker and worker._stop_requested:
                        logging.info(f"Clustering cancelled during distance calculations at batch {batch_start}-{batch_end}")
                        return None, None

                    for i in range(batch_start, batch_end):
                        cluster_idx = labels[i]
                        if cluster_idx >= 0:  # Skip noise points
                            center = clusterer.cluster_centers_[cluster_idx]
                            distances[i] = np.linalg.norm(clean_data[i] - center)

                    # Update progress during batch processing
                    if worker:
                        progress = 80 + int((batch_end / len(clean_data)) * 10)
                        worker.progress_updated.emit(progress)

                # Normalize distances to [0, 1] range and invert (closer = higher probability)
                if len(distances) > 0:
                    max_dist = np.max(distances) if np.max(distances) > 0 else 1
                    probabilities = 1 - (distances / max_dist)
                else:
                    probabilities = np.array([])

                # Create full-sized array for probabilities
                full_probabilities = np.zeros(len(data))
                full_probabilities[mask] = probabilities

            else:
                logging.error(f"Unsupported clustering method: {method}")
                return None, None

            # Report progress: 90% - Updating data frame
            if worker:
                worker.progress_updated.emit(90)
                # Check if stop was requested
                if worker._stop_requested:
                    logging.info("Clustering cancelled before data frame update")
                    return None, None

            # Report progress: 100% - Clustering completed
            if worker:
                worker.progress_updated.emit(100)

            # Store the data shape used for clustering
            self._cluster_data_shape = len(data)
            logging.log(0, f"Stored cluster data shape: {self._cluster_data_shape}")

            return full_labels, full_probabilities

        except Exception as e:
            logging.error(f"Error during {method} clustering: {str(e)}")
            return None, None
