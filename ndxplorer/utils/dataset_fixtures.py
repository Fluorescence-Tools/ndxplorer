"""Synthetic dataset utilities for integration and export testing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from ..logging_config import logging
from ..export.api import save_selection
from ..export.models import SelectionExportPayload

try:  # pragma: no cover - optional dependency in test envs
    import matplotlib

    matplotlib.use("Agg", force=True)  # Always render offscreen
    from matplotlib import pyplot as plt

    _MATPLOTLIB_AVAILABLE = True
except Exception:  # pragma: no cover - fallback when matplotlib absent
    plt = None
    _MATPLOTLIB_AVAILABLE = False


_DEFAULT_COLUMNS = (
    "burst_duration_ns",
    "transfer_efficiency",
    "burst_size",
    "photon_count",
)


@dataclass
class FixtureSummary:
    """Bookkeeping info returned after writing a dataset bundle."""

    csv_path: Path
    hdf5_path: Path
    image_path: Optional[Path]
    npz_path: Path
    metadata_path: Path
    corrupted_manifest: Optional[Path]

    def as_dict(self) -> Dict[str, Optional[str]]:
        return {
            "csv": str(self.csv_path),
            "hdf5": str(self.hdf5_path),
            "image": str(self.image_path) if self.image_path else None,
            "npz": str(self.npz_path),
            "metadata": str(self.metadata_path),
            "corrupted_manifest": str(self.corrupted_manifest)
            if self.corrupted_manifest
            else None,
        }


def _generate_feature_matrix(
    n_points: int,
    seed: int,
    columns: Iterable[str],
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cols = list(columns)
    if len(cols) < 2:
        raise ValueError("At least two columns must be provided for synthetic data.")

    duration = rng.gamma(shape=2.0, scale=2.5, size=n_points)
    efficiency = np.clip(rng.normal(loc=0.4, scale=0.2, size=n_points), 0, 1)
    burst_size = rng.lognormal(mean=0.0, sigma=0.75, size=n_points)
    photon_count = rng.poisson(lam=50, size=n_points).astype(np.float64)

    generators = [duration, efficiency, burst_size, photon_count]
    while len(generators) < len(cols):
        generators.append(rng.normal(size=n_points))

    matrix = np.vstack(generators[: len(cols)]).astype(np.float64)
    return matrix


def _build_selection_payload(
    *,
    values: np.ndarray,
    columns: List[str],
    seed: int,
    include_figure: bool,
    name: Optional[str] = None,
) -> SelectionExportPayload:
    metadata = {
        "source": "ndxplorer.synthetic",
        "seed": seed,
        "columns": columns,
        "n_points": int(values.shape[1]),
    }
    figure = None
    if include_figure and _MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.hexbin(values[0], values[1], gridsize=80, cmap="viridis")
        ax.set_xlabel(columns[0])
        ax.set_ylabel(columns[1])
        ax.set_title("Synthetic Selection Preview")
        fig.tight_layout()
        figure = fig
    elif include_figure:
        logging.warning(
            "Matplotlib unavailable – dataset figure export will be skipped."
        )

    payload = SelectionExportPayload(
        selections=[],
        values=values,
        columns=columns,
        metadata=metadata,
        name=name or f"synthetic_{values.shape[1]}",
        figure=figure,
    )
    return payload


def create_synthetic_selection_payload(
    *,
    n_points: int = 1_000_000,
    seed: int = 13,
    columns: Optional[Iterable[str]] = None,
    include_figure: bool = True,
) -> SelectionExportPayload:
    """
    Build a SelectionExportPayload populated with synthetic burst data.
    """

    columns = list(columns) if columns is not None else list(_DEFAULT_COLUMNS)
    if not columns:
        raise ValueError("Synthetic payload requires at least one column name.")

    values = _generate_feature_matrix(n_points=n_points, seed=seed, columns=columns)
    payload = _build_selection_payload(
        values=values,
        columns=columns,
        seed=seed,
        include_figure=include_figure,
        name=f"synthetic_{n_points}_points",
    )
    return payload


def _create_corrupted_manifest(payload: SelectionExportPayload) -> Dict[str, object]:
    truncated = payload.values[:-1] if payload.values is not None else None
    return {
        "description": "Corrupted payload missing final feature row.",
        "values_shape": truncated.shape if truncated is not None else None,
        "columns": payload.columns,
        "metadata": payload.metadata,
    }


def write_fixture_bundle(
    output_dir: Path | str,
    *,
    n_points: int = 1_000_000,
    seed: int = 13,
    columns: Optional[Iterable[str]] = None,
    include_corrupted: bool = True,
) -> FixtureSummary:
    """
    Materialize CSV/HDF5/image fixtures plus raw arrays for cross-team testing.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = create_synthetic_selection_payload(
        n_points=n_points, seed=seed, columns=columns, include_figure=True
    )

    csv_path = output_dir / "ndx_selection.csv"
    hdf5_path = output_dir / "ndx_selection.h5"
    image_path = output_dir / "ndx_selection.png"
    npz_path = output_dir / "ndx_selection_values.npz"
    metadata_path = output_dir / "ndx_selection_metadata.json"

    save_selection(payload, csv_path, format="csv")
    save_selection(payload, hdf5_path, format="hdf5")
    if payload.has_drawable():
        save_selection(payload, image_path, format="image")
    else:
        image_path = None

    np.savez_compressed(npz_path, values=payload.values, columns=np.array(payload.columns))
    metadata_path.write_text(payload.to_json_metadata(), encoding="utf-8")

    corrupted_manifest_path = None
    if include_corrupted:
        manifest = _create_corrupted_manifest(payload)
        corrupted_manifest_path = output_dir / "corrupted_payload.json"
        corrupted_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if payload.figure is not None and _MATPLOTLIB_AVAILABLE:
        plt.close(payload.figure)

    return FixtureSummary(
        csv_path=csv_path,
        hdf5_path=hdf5_path,
        image_path=image_path,
        npz_path=npz_path,
        metadata_path=metadata_path,
        corrupted_manifest=corrupted_manifest_path,
    )


def main(argv: Optional[List[str]] = None) -> Dict[str, Optional[str]]:
    """CLI interface for dataset bundle generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate NDxplorer synthetic datasets.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination directory.")
    parser.add_argument("--n-points", type=int, default=1_000_000, help="Number of synthetic points.")
    parser.add_argument("--seed", type=int, default=13, help="RNG seed.")
    parser.add_argument(
        "--no-corrupted",
        action="store_true",
        help="Skip writing corrupted payload manifest.",
    )
    args = parser.parse_args(argv)

    summary = write_fixture_bundle(
        output_dir=args.output_dir,
        n_points=args.n_points,
        seed=args.seed,
        include_corrupted=not args.no_corrupted,
    )
    as_dict = summary.as_dict()
    print(json.dumps(as_dict, indent=2))
    return as_dict


if __name__ == "__main__":  # pragma: no cover - CLI passthrough
    main()
