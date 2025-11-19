"""
Centralized lazy import helpers for heavy optional dependencies used by NDxplorer.
Each getter attempts to import on first use, caches the result, and returns
None if the package is unavailable. Minimal, concise logging is performed on
first successful import or when a dependency is missing.
"""
from __future__ import annotations

from typing import Optional, Any

from .logging_config import logging

# Caches
__umap: Optional[Any] = None
__kmeans_cls: Optional[Any] = None
__gmm_cls: Optional[Any] = None
__hdbscan: Optional[Any] = None
__napari: Optional[Any] = None


def get_umap():
    global __umap
    if __umap is not None:
        return __umap
    try:
        import umap as _umap  # type: ignore
        __umap = _umap
        logging.debug("lazy_imports: umap imported on demand")
    except Exception:
        __umap = None
        logging.debug("UMAP not available")
    return __umap


def get_kmeans():
    global __kmeans_cls
    if __kmeans_cls is not None:
        return __kmeans_cls
    try:
        from sklearn.cluster import KMeans as _KMeans  # type: ignore
        __kmeans_cls = _KMeans
        logging.debug("lazy_imports: sklearn KMeans imported on demand")
    except Exception:
        __kmeans_cls = None
        logging.debug("scikit-learn not available")
    return __kmeans_cls


def get_gmm():
    global __gmm_cls
    if __gmm_cls is not None:
        return __gmm_cls
    try:
        from sklearn.mixture import GaussianMixture as _GMM  # type: ignore
        __gmm_cls = _GMM
        logging.debug("lazy_imports: sklearn GaussianMixture imported on demand")
    except Exception:
        __gmm_cls = None
        logging.debug("scikit-learn not available")
    return __gmm_cls


def get_hdbscan():
    global __hdbscan
    if __hdbscan is not None:
        return __hdbscan
    try:
        import hdbscan as _hdbscan  # type: ignore
        __hdbscan = _hdbscan
        logging.debug("lazy_imports: hdbscan imported on demand")
    except Exception:
        __hdbscan = None
        logging.debug("HDBSCAN not available")
    return __hdbscan


def get_napari():
    global __napari
    if __napari is not None:
        return __napari
    try:
        import napari as _napari  # type: ignore
        __napari = _napari
        logging.debug("lazy_imports: napari imported on demand")
    except Exception:
        __napari = None
        logging.debug("napari not available")
    return __napari
