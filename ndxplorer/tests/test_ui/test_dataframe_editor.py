"""Tests for the DataFrameEditor widget."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from qtpy import QtCore, QtWidgets

from ndxplorer.ui.dataframe_editor import DataFrameEditor


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "float_col": [1.0, 2.5, np.nan, 4.2],
        "int_col": [10, 20, 30, 40],
        "str_col": ["a", "b", "c", "d"],
    })


def test_create_and_title(qapp: QtWidgets.QApplication, sample_df: pd.DataFrame):
    dlg = DataFrameEditor(sample_df, None)
    assert dlg.windowTitle() == "DataFrame Editor"
    assert dlg.dataframe.shape == (4, 3)


def test_edit_numeric_cell(qapp: QtWidgets.QApplication, sample_df: pd.DataFrame):
    dlg = DataFrameEditor(sample_df, None)
    item = dlg._table.item(0, 0)
    assert item is not None
    assert "1" in item.text()


def test_search_filters_rows(qapp: QtWidgets.QApplication, sample_df: pd.DataFrame):
    dlg = DataFrameEditor(sample_df, None)
    assert dlg._table.rowCount() == 4
    dlg._search_edit.setText("2.5")
    assert dlg._table.rowCount() == 1


def test_nan_shown_as_empty(qapp: QtWidgets.QApplication, sample_df: pd.DataFrame):
    dlg = DataFrameEditor(sample_df, None)
    item = dlg._table.item(2, 0)
    assert item is not None
    assert item.text() == ""


def test_column_count_matches(qapp: QtWidgets.QApplication, sample_df: pd.DataFrame):
    dlg = DataFrameEditor(sample_df, None)
    assert dlg._table.columnCount() == 3
