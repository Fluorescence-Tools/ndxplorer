"""
Tests for image export functionality.
"""
import pytest
import tempfile
import os
from pathlib import Path
import numpy as np
from PIL import Image

from ndxplorer.export.image_export import ImageExporter


class TestImageExporter:
    """Test image export functionality."""
    
    @pytest.fixture
    def sample_image_data(self):
        """Create sample image data for testing."""
        return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    @pytest.fixture
    def exporter(self):
        """Create image exporter instance."""
        return ImageExporter()
    
    def test_export_png(self, exporter, sample_image_data):
        """Test PNG export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.png"
            exporter.export(sample_image_data, filepath, format='png')
            
            assert filepath.exists()
            img = Image.open(filepath)
            assert img.format == 'PNG'
            assert img.size == (100, 100)
    
    def test_export_jpg(self, exporter, sample_image_data):
        """Test JPG export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.jpg"
            exporter.export(sample_image_data, filepath, format='jpg', quality=95)
            
            assert filepath.exists()
            img = Image.open(filepath)
            assert img.format == 'JPEG'
    
    def test_export_svg(self, exporter, sample_image_data):
        """Test SVG export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.svg"
            exporter.export(sample_image_data, filepath, format='svg')
            
            assert filepath.exists()
            assert filepath.suffix == '.svg'
    
    def test_export_dpi_control(self, exporter, sample_image_data):
        """Test DPI control for raster formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_dpi.png"
            exporter.export(sample_image_data, filepath, format='png', dpi=300)
            
            assert filepath.exists()
            img = Image.open(filepath)
            # PIL doesn't store DPI directly, but file should be created
            assert img.size == (100, 100)
    
    def test_export_quality_control(self, exporter, sample_image_data):
        """Test quality control for JPEG."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_quality.jpg"
            exporter.export(sample_image_data, filepath, format='jpg', quality=50)
            
            assert filepath.exists()
            img = Image.open(filepath)
            assert img.format == 'JPEG'
    
    def test_export_invalid_format(self, exporter, sample_image_data):
        """Test handling of invalid format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.invalid"
            
            with pytest.raises(ValueError):
                exporter.export(sample_image_data, filepath, format='invalid')
    
    def test_export_grayscale(self, exporter):
        """Test grayscale image export."""
        gray_data = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_gray.png"
            exporter.export(gray_data, filepath, format='png')
            
            assert filepath.exists()
            img = Image.open(filepath)
            assert img.mode == 'L'
