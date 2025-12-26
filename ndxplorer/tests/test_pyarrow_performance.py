"""
Tests for PyArrow performance optimizations in NDXplorer.

This module tests:
1. ArrowDataSource functionality
2. Fast CSV reading with PyArrow
3. Fast numeric conversion
4. Performance benchmarks
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import time

# Check if PyArrow is available
try:
    import pyarrow as pa
    HAVE_PYARROW = True
except ImportError:
    HAVE_PYARROW = False


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n_rows = 10000
    return pd.DataFrame({
        'col_a': np.random.randn(n_rows),
        'col_b': np.random.randn(n_rows),
        'col_c': np.random.randint(0, 100, n_rows),
        'col_d': np.random.choice(['a', 'b', 'c'], n_rows),
        'col_e': np.random.randn(n_rows).astype(str),  # String numbers
    })


@pytest.fixture
def large_csv_file(tmp_path):
    """Create a large CSV file for benchmarking."""
    np.random.seed(42)
    n_rows = 50000
    n_cols = 20
    
    data = {f'col_{i}': np.random.randn(n_rows) for i in range(n_cols)}
    df = pd.DataFrame(data)
    
    csv_path = tmp_path / "large_test.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


class TestFastNumericConversion:
    """Test the _fast_to_numeric function."""
    
    def test_numeric_passthrough(self, sample_dataframe):
        """Numeric columns should pass through unchanged."""
        from ndxplorer.core.data_source import _fast_to_numeric
        
        df = sample_dataframe[['col_a', 'col_b', 'col_c']].copy()
        result = _fast_to_numeric(df)
        
        assert result.shape == df.shape
        assert all(pd.api.types.is_numeric_dtype(result[col]) for col in result.columns)
    
    def test_string_conversion(self, sample_dataframe):
        """String numeric columns should be converted."""
        from ndxplorer.core.data_source import _fast_to_numeric
        
        df = sample_dataframe[['col_e']].copy()
        result = _fast_to_numeric(df)
        
        assert pd.api.types.is_numeric_dtype(result['col_e'])
    
    def test_mixed_types(self, sample_dataframe):
        """Mixed type DataFrames should be handled."""
        from ndxplorer.core.data_source import _fast_to_numeric
        
        result = _fast_to_numeric(sample_dataframe)
        
        assert result.shape == sample_dataframe.shape
        # Non-numeric string column should become NaN
        assert result['col_d'].isna().all()
    
    def test_empty_dataframe(self):
        """Empty DataFrames should be handled."""
        from ndxplorer.core.data_source import _fast_to_numeric
        
        df = pd.DataFrame()
        result = _fast_to_numeric(df)
        
        assert result.empty


class TestDataSource:
    """Test DataSource with PyArrow optimizations."""
    
    def test_data_assignment(self, sample_dataframe):
        """Test that data assignment works with PyArrow optimization."""
        from ndxplorer.core.data_source import DataSource
        
        ds = DataSource(data=sample_dataframe)
        
        assert not ds.empty
        assert ds.size == len(sample_dataframe)
        assert len(ds.parameter_names) == len(sample_dataframe.columns)
    
    def test_values_property(self, sample_dataframe):
        """Test that values property returns correct shape."""
        from ndxplorer.core.data_source import DataSource
        
        ds = DataSource(data=sample_dataframe[['col_a', 'col_b', 'col_c']])
        values = ds.values
        
        assert values.shape == (3, len(sample_dataframe))
        assert values.dtype == np.float32


@pytest.mark.skipif(not HAVE_PYARROW, reason="PyArrow not available")
class TestArrowDataSource:
    """Test ArrowDataSource functionality."""
    
    def test_from_pandas(self, sample_dataframe):
        """Test creating ArrowDataSource from pandas DataFrame."""
        from ndxplorer.core.arrow_backend import ArrowDataSource
        
        ds = ArrowDataSource.from_pandas(sample_dataframe)
        
        assert not ds.empty
        assert ds.size == len(sample_dataframe)
    
    def test_values_property(self, sample_dataframe):
        """Test values property returns correct array."""
        from ndxplorer.core.arrow_backend import ArrowDataSource
        
        ds = ArrowDataSource.from_pandas(sample_dataframe[['col_a', 'col_b']])
        values = ds.values
        
        assert values.shape == (2, len(sample_dataframe))
        assert values.dtype == np.float32
    
    def test_get_column(self, sample_dataframe):
        """Test column retrieval."""
        from ndxplorer.core.arrow_backend import ArrowDataSource
        
        ds = ArrowDataSource.from_pandas(sample_dataframe)
        
        col_a = ds.get_column('col_a')
        assert len(col_a) == len(sample_dataframe)
        
        col_0 = ds.get_column(0)
        assert len(col_0) == len(sample_dataframe)
    
    def test_merge_columns(self, sample_dataframe):
        """Test column merge."""
        from ndxplorer.core.arrow_backend import ArrowDataSource
        
        ds1 = ArrowDataSource.from_pandas(sample_dataframe[['col_a', 'col_b']])
        ds2 = ArrowDataSource.from_pandas(sample_dataframe[['col_c']])
        
        result = ds1.merge(ds2, mode='columns')
        
        assert result
        assert len(ds1.parameter_names) == 3
    
    def test_from_csv(self, large_csv_file):
        """Test CSV reading with ArrowDataSource."""
        from ndxplorer.core.arrow_backend import ArrowDataSource
        
        ds = ArrowDataSource.from_csv(large_csv_file)
        
        assert not ds.empty
        assert ds.size == 50000


@pytest.mark.skipif(not HAVE_PYARROW, reason="PyArrow not available")
class TestPyArrowCSVReader:
    """Test PyArrow CSV reader functions."""
    
    def test_read_csv_pyarrow(self, large_csv_file):
        """Test read_csv_pyarrow function."""
        from ndxplorer.io.reader import read_csv_pyarrow
        
        df = read_csv_pyarrow(large_csv_file)
        
        assert len(df) == 50000
        assert len(df.columns) == 20
    
    def test_read_csv_fast(self, large_csv_file):
        """Test read_csv_fast function."""
        from ndxplorer.io.reader import read_csv_fast
        
        df = read_csv_fast(large_csv_file)
        
        assert len(df) == 50000


@pytest.mark.benchmark
@pytest.mark.skipif(not HAVE_PYARROW, reason="PyArrow not available")
class TestPerformanceBenchmarks:
    """Performance benchmarks comparing pandas vs PyArrow."""
    
    def test_csv_read_benchmark(self, large_csv_file):
        """Benchmark CSV reading: PyArrow vs pandas."""
        from ndxplorer.io.reader import read_csv_pyarrow
        
        # PyArrow
        t0 = time.perf_counter()
        df_arrow = read_csv_pyarrow(large_csv_file)
        t_arrow = time.perf_counter() - t0
        
        # Pandas
        t0 = time.perf_counter()
        df_pandas = pd.read_csv(large_csv_file)
        t_pandas = time.perf_counter() - t0
        
        print(f"\nCSV Read Benchmark (50k rows, 20 cols):")
        print(f"  PyArrow: {t_arrow:.3f}s")
        print(f"  Pandas:  {t_pandas:.3f}s")
        print(f"  Speedup: {t_pandas/t_arrow:.1f}x")
        
        assert len(df_arrow) == len(df_pandas)
    
    def test_numeric_conversion_benchmark(self):
        """Benchmark numeric conversion: PyArrow vs pandas."""
        from ndxplorer.core.data_source import _fast_to_numeric, _HAVE_PYARROW
        
        # Create mixed-type DataFrame
        np.random.seed(42)
        n_rows = 100000
        df = pd.DataFrame({
            'float_col': np.random.randn(n_rows),
            'int_col': np.random.randint(0, 1000, n_rows),
            'str_num_col': np.random.randn(n_rows).astype(str),
            'str_col': np.random.choice(['a', 'b', 'c'], n_rows),
        })
        
        # Fast conversion (uses PyArrow if available)
        t0 = time.perf_counter()
        result_fast = _fast_to_numeric(df.copy())
        t_fast = time.perf_counter() - t0
        
        # Pandas apply
        t0 = time.perf_counter()
        result_pandas = df.copy().apply(pd.to_numeric, errors='coerce')
        t_pandas = time.perf_counter() - t0
        
        print(f"\nNumeric Conversion Benchmark (100k rows, 4 cols):")
        print(f"  Fast (PyArrow={_HAVE_PYARROW}): {t_fast:.3f}s")
        print(f"  Pandas apply: {t_pandas:.3f}s")
        print(f"  Speedup: {t_pandas/t_fast:.1f}x")
        
        assert result_fast.shape == result_pandas.shape
    
    def test_datasource_initialization_benchmark(self):
        """Benchmark DataSource initialization."""
        from ndxplorer.core.data_source import DataSource
        
        np.random.seed(42)
        n_rows = 100000
        n_cols = 50
        
        df = pd.DataFrame(
            np.random.randn(n_rows, n_cols),
            columns=[f'col_{i}' for i in range(n_cols)]
        )
        
        t0 = time.perf_counter()
        ds = DataSource(data=df)
        t_init = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        _ = ds.values
        t_values = time.perf_counter() - t0
        
        print(f"\nDataSource Initialization Benchmark (100k rows, 50 cols):")
        print(f"  Initialization: {t_init:.3f}s")
        print(f"  Values access:  {t_values:.3f}s")
        
        assert ds.size == n_rows


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s', '-k', 'benchmark'])
