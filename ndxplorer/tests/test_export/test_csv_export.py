"""
Tests for CSV export functionality.
"""
import pytest
import tempfile
import os
import pandas as pd
import numpy as np
from pathlib import Path

from ndxplorer.export.csv_export import CSVExporter


class TestCSVExporter:
    """Test CSV export functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'x': np.arange(10),
            'y': np.random.random(10),
            'category': ['A', 'B'] * 5
        })
    
    @pytest.fixture
    def exporter(self):
        """Create CSV exporter instance."""
        return CSVExporter()
    
    def test_export_basic_csv(self, exporter, sample_data):
        """Test basic CSV export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.csv"
            exporter.export(sample_data, filepath)
            
            assert filepath.exists()
            loaded = pd.read_csv(filepath)
            pd.testing.assert_frame_equal(loaded, sample_data)
    
    def test_export_tsv(self, exporter, sample_data):
        """Test TSV export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.tsv"
            exporter.export(sample_data, filepath, delimiter='\t')
            
            assert filepath.exists()
            loaded = pd.read_csv(filepath, sep='\t')
            pd.testing.assert_frame_equal(loaded, sample_data)
    
    def test_export_with_metadata(self, exporter, sample_data):
        """Test CSV export with metadata."""
        metadata = {"experiment": "test", "date": "2025-12-25"}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.csv"
            exporter.export(sample_data, filepath, metadata=metadata)
            
            assert filepath.exists()
            
            # Check metadata file
            meta_file = filepath.with_suffix('.csv.meta')
            assert meta_file.exists()
    
    def test_export_empty_dataframe(self, exporter):
        """Test export with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "empty.csv"
            exporter.export(empty_df, filepath)
            
            assert filepath.exists()
            loaded = pd.read_csv(filepath)
            assert len(loaded) == 0
    
    def test_export_large_dataset(self, exporter):
        """Test export performance with large dataset."""
        large_data = pd.DataFrame({
            'x': np.arange(10000),
            'y': np.random.random(10000),
            'z': np.random.random(10000)
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "large.csv"
            exporter.export(large_data, filepath)
            
            assert filepath.exists()
            loaded = pd.read_csv(filepath)
            pd.testing.assert_frame_equal(loaded, large_data)
