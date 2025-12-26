"""
Basic tests for UI components.
"""
import pytest
from PyQt6.QtWidgets import QApplication
from ndxplorer.ui import histogram_controls, selection_panel, parameter_editor


@pytest.fixture(scope="session")
def qapp():
    """Create QApplication instance for tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestUIComponents:
    """Test UI component initialization."""
    
    def test_histogram_controls_creation(self, qapp):
        """Test histogram controls widget creation."""
        widget = histogram_controls.HistogramControls()
        assert widget is not None
    
    def test_selection_panel_creation(self, qapp):
        """Test selection panel widget creation."""
        widget = selection_panel.SelectionPanel()
        assert widget is not None
    
    def test_parameter_editor_creation(self, qapp):
        """Test parameter editor widget creation."""
        widget = parameter_editor.ParameterEditor()
        assert widget is not None
