"""
Basic tests for UI components.
"""
import pytest
from qtpy.QtWidgets import QApplication
from ndxplorer.ui import parameter_editor


@pytest.fixture(scope="session")
def qapp():
    """Create QApplication instance for tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestUIComponents:
    """Test UI component initialization."""
    
    def test_parameter_editor_creation(self, qapp):
        """Test parameter editor widget creation."""
        widget = parameter_editor.ParameterEditor()
        assert widget is not None
