"""Helpers for preparing export file paths across platforms."""

from __future__ import annotations

from pathlib import Path
from typing import Union
import os
import logging

IS_WINDOWS = os.name == "nt"
LONG_PATH_THRESHOLD = 240  # Windows default MAX_PATH (260) minus allowance for prefix
LOG = logging.getLogger(__name__)


def normalize_export_path(path: Union[str, Path]) -> Path:
    """
    Return a pathlib.Path that is safe to use for exports across platforms.

    Steps:
    - Expand user home markers (``~``)
    - Resolve relative paths without requiring that the destination already exists
    - On Windows, switch to long-path / UNC-aware syntax when necessary
    """

    expanded = Path(path).expanduser()
    try:
        resolved = expanded.resolve(strict=False)
    except RuntimeError:
        resolved = expanded.absolute()

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        LOG.warning("Could not create parent directory '%s': %s", resolved.parent, exc)

    if not IS_WINDOWS:
        return resolved

    raw = str(resolved)
    # Already normalized
    if raw.startswith("\\\\?\\"):
        return Path(raw)

    # UNC share: \\server\share -> \\?\UNC\server\share
    if raw.startswith("\\\\"):
        unc_body = raw[2:]
        return Path(f"\\\\?\\UNC\\{unc_body}")

    if len(raw) >= LONG_PATH_THRESHOLD:
        return Path(f"\\\\?\\{raw}")

    return resolved


__all__ = ["normalize_export_path", "IS_WINDOWS", "LONG_PATH_THRESHOLD"]
