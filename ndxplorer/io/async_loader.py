"""Utilities for running expensive data-loading operations off the UI thread."""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Callable, Optional

from qtpy import QtCore

from ..logging_config import logging

if False:  # pragma: no cover - for type checkers only
    from ..core.data_source import DataSource


@dataclass
class DataLoadTask:
    """Description of a deferred data loading job."""

    description: str
    load_callable: Callable[[], "DataSource"]
    on_success: Callable[["DataSource"], None]
    on_error: Optional[Callable[[str], None]] = None


@dataclass
class DataLoadResult:
    """Bundle emitted when the worker finishes successfully."""

    task: DataLoadTask
    data_source: "DataSource"


class DataLoadWorker(QtCore.QObject):
    """Qt worker object that executes a :class:`DataLoadTask` in a thread."""

    finished = QtCore.Signal(object)  # DataLoadResult
    error = QtCore.Signal(str)

    def __init__(self, task: DataLoadTask):
        super().__init__()
        self._task = task

    @QtCore.Slot()
    def run(self) -> None:
        try:
            logging.info("Starting background data load: %s", self._task.description)
            data_source = self._task.load_callable()
        except Exception:  # pragma: no cover - GUI path
            logging.exception("Background data load failed")
            self.error.emit(traceback.format_exc())
            return

        self.finished.emit(DataLoadResult(task=self._task, data_source=data_source))


def run_task_inline(task: DataLoadTask) -> None:
    """Execute ``task`` synchronously (used when no GUI thread is available)."""

    try:
        data_source = task.load_callable()
    except Exception as exc:  # pragma: no cover - CLI/tests
        logging.exception("Synchronous data load failed")
        if task.on_error is not None:
            task.on_error(str(exc))
        else:
            raise
        return

    task.on_success(data_source)
