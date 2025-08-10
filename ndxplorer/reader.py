from typing import List, Union, Optional, BinaryIO, TextIO
import pathlib
import os
import tempfile
import zipfile
import io
import shutil

try:
    from chisurf import logging
except ImportError:
    import logging
import pandas as pd
from pandas.errors import EmptyDataError

from . data_source import DataSource

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QProgressBar, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QCoreApplication

"""
NDXplorer Reader Module

This module provides functions for reading various data formats for the NDXplorer application.
It supports:
- CSV files (regular and zipped)
- MFD HDF5 files (regular and zipped)
- Burst analysis directories (regular and zipped)

For zipped CSV files, pandas is used to read the data directly from the zip archive.
For zipped MFD folders, the zip is extracted to a temporary directory before processing.
For zipped HDF5 files, the file is extracted to a temporary location before reading.
"""


# reader.py (top-level helpers)
def _zip_contains_any(zip_path: str, exts: tuple) -> bool:
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            names = [n.lower() for n in zf.namelist()]
        return any(n.endswith(ext) for ext in exts for n in names)
    except Exception as e:
        try:
            logging.debug(f"Zip inspect failed for '{zip_path}': {e}")
        except Exception:
            pass
        return False


class ProgressWindow(QDialog):
    def __init__(self, title="Progress", message="Processing...", max_value=100, parent=None):
        super().__init__(parent)

        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModal)  # Keep this window in front

        self.layout = QVBoxLayout()
        self.label = QLabel(message)
        self.progress_bar = QProgressBar()

        # Set the range of the progress bar (0 to max_value)
        self.progress_bar.setRange(0, max_value)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.progress_bar)
        self.setLayout(self.layout)

    def set_value(self, value: int):
        """
        Update the progress bar to the given value.
        """
        self.progress_bar.setValue(value)


def read_burst_analysis(
        base_path: Union[str, pathlib.Path] = "./test/mfd/burstwise_All 0.2500#30",
        skip_nth_row: int = 2,
        additional_endings: List[str] = None,
        drop_last_column: bool = True
) -> DataSource:
    """
    Reads .bur files and any additional files specified,
    including files that have only headers or are completely empty.
    Column names are preserved exactly as in the source files.

    Combines "Mean Macro Time (ms)" values sequentially across files,
    converts them to seconds, and renames the column to "Mean Macro Time (s)".
    
    Supports multiple input formats:
    1. Regular directories with bi4_bur or bur subdirectories
    2. Zipped MFD folders with the standard directory structure
    3. Zip files created by the burst selector, which may have .bur files directly in the zip
       or in various subdirectories
    
    For zip files, the function first tries to find .bur files directly in the zip.
    If found, it extracts them to a temporary directory with the expected structure.
    If no .bur files are found or if direct processing fails, it falls back to
    extracting the entire zip file and looking for the standard directory structure.
    """
    # ensure a QApplication
    app = QApplication.instance() or QApplication([])
    base_path = pathlib.Path(base_path)
    additional_endings = additional_endings or ["bg4", "br4", "by4", "bv4"]
    
    # Check if the path is a zip file
    if base_path.is_file() and base_path.suffix.lower() == '.zip':
        # First, try to process the zip file directly to see if it contains .bur files
        # This is for zip files created by the burst selector
        try:
            with zipfile.ZipFile(base_path, 'r') as zip_file:
                # Get all files in the zip
                all_files = zip_file.namelist()
                print(f"Files in zip: {all_files}")
                
                # Look for .bur files in the zip
                bur_files = [f for f in all_files if f.endswith('.bur')]
                print(f"Found .bur files: {bur_files}")
                
                # If we found .bur files, process them directly from the zip
                if bur_files:
                    # Create a temporary directory to extract only the .bur files
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_path = pathlib.Path(temp_dir)
                        
                        # Create bi4_bur directory in the temp directory
                        bur_dir = temp_path / "bi4_bur"
                        bur_dir.mkdir(exist_ok=True)
                        
                        # Extract all .bur files to the bi4_bur directory
                        for bur_file in bur_files:
                            try:
                                # Extract the file
                                print(f"Extracting {bur_file} to {temp_path}")
                                zip_file.extract(bur_file, temp_path)
                                
                                # Move the file to the bi4_bur directory if it's not already there
                                extracted_path = temp_path / bur_file
                                print(f"Extracted path: {extracted_path}, bur_dir: {bur_dir}")
                                if extracted_path.parent != bur_dir:
                                    # Make sure parent directories exist
                                    bur_dir.mkdir(exist_ok=True, parents=True)
                                    
                                    # Move the file
                                    target_path = bur_dir / extracted_path.name
                                    print(f"Moving {extracted_path} to {target_path}")
                                    shutil.move(str(extracted_path), str(target_path))
                            except Exception as e:
                                print(f"Error processing {bur_file}: {str(e)}")
                        
                        # Extract any additional files if they exist
                        for ending in additional_endings:
                            additional_files = [f for f in all_files if f.endswith(f'.{ending}')]
                            print(f"Found additional files for ending '{ending}': {additional_files}")
                            if additional_files:
                                try:
                                    # Create directory for this ending
                                    ending_dir = temp_path / ending
                                    ending_dir.mkdir(exist_ok=True)
                                    print(f"Created directory for {ending}: {ending_dir}")
                                    
                                    # Extract and move files
                                    for add_file in additional_files:
                                        try:
                                            print(f"Extracting additional file {add_file} to {temp_path}")
                                            zip_file.extract(add_file, temp_path)
                                            extracted_path = temp_path / add_file
                                            if extracted_path.parent != ending_dir:
                                                target_path = ending_dir / extracted_path.name
                                                print(f"Moving additional file {extracted_path} to {target_path}")
                                                shutil.move(str(extracted_path), str(target_path))
                                        except Exception as e:
                                            print(f"Error processing additional file {add_file}: {str(e)}")
                                except Exception as e:
                                    print(f"Error processing files with ending '{ending}': {str(e)}")
                        
                        # Process the temporary directory
                        return _process_burst_analysis_dir(
                            temp_path,
                            skip_nth_row,
                            additional_endings,
                            drop_last_column
                        )
        except Exception as e:
            # If direct processing fails, fall back to the original method
            print(f"Direct zip processing failed: {str(e)}. Falling back to extraction method.")
        
        # If we get here, either no .bur files were found or an error occurred
        # Fall back to the original method of extracting the entire zip
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            
            # Extract the zip file
            with zipfile.ZipFile(base_path, 'r') as zip_file:
                zip_file.extractall(temp_path)
            
            # Find the MFD folder in the extracted contents
            # Look for directories that might contain burst analysis data
            potential_dirs = [d for d in temp_path.iterdir() if d.is_dir()]
            
            # If there's only one directory, use it
            if len(potential_dirs) == 1:
                mfd_dir = potential_dirs[0]
            else:
                # Try to find a directory with bi4_bur, bur, or hdf5 subdirectories
                mfd_dir = None
                for d in potential_dirs:
                    if (d / "bi4_bur").exists() or (d / "bur").exists() or (d / "hdf5").exists():
                        mfd_dir = d
                        break
                
                # If still not found, use the temp directory itself
                if mfd_dir is None:
                    mfd_dir = temp_path
            
            # Now process the extracted directory
            return _process_burst_analysis_dir(
                mfd_dir, 
                skip_nth_row, 
                additional_endings, 
                drop_last_column
            )
    else:
        # Regular directory processing
        return _process_burst_analysis_dir(
            base_path, 
            skip_nth_row, 
            additional_endings, 
            drop_last_column
        )


def _process_burst_analysis_dir(
        base_path: pathlib.Path,
        skip_nth_row: int = 2,
        additional_endings: List[str] = None,
        drop_last_column: bool = True
) -> DataSource:
    """
    Process a burst analysis directory (helper function for read_burst_analysis).
    Prefer HDF5 if an 'hdf5' subfolder with .h5/.hdf5 exists; otherwise, read BUR files.
    """
    # Prefer HDF5 if available
    hdf5_dir = base_path / "hdf5"
    if hdf5_dir.is_dir():
        hdf5_files = sorted([p for p in hdf5_dir.iterdir() if p.suffix.lower() in (".h5", ".hdf5")])
        if hdf5_files:
            logging.info(f"NDXplorer: Found HDF5 folder, reading {hdf5_files[0]}")
            # Use existing HDF5 reader path
            return read_mfd_hdf5([str(hdf5_files[0])])

    # locate .bur files
    dir_main = base_path / "bi4_bur"
    dir_fallback = base_path / "bur"
    if dir_main.is_dir():
        bur_files = list(dir_main.glob("*.bur")) or list(dir_fallback.glob("*.bur"))
    else:
        bur_files = list(dir_fallback.glob("*.bur"))
    if not bur_files:
        raise FileNotFoundError("No .bur files found in either 'bi4_bur' or 'bur'.")

    # helper: read CSV or build header‐only DataFrame
    def _read_file(path: pathlib.Path) -> pd.DataFrame:
        try:
            df = pd.read_csv(path, sep="\t")
        except EmptyDataError:
            # try to pull headers even if there's no data rows
            with open(path, 'r') as f:
                first = f.readline().strip()
            cols = first.split("\t") if first else []
            df = pd.DataFrame(columns=cols)
        return df

    # progress dialog
    progress = ProgressWindow(
        title="File Processing",
        message="Processing Burst files...",
        max_value=len(bur_files),
    )
    progress.show()

    pieces = []
    macro_time_offset = 0.0  # Keep track of the cumulative macro time offset
    macro_time_column = "Mean Macro Time (ms)"
    macro_time_column_seconds = "Mean Macro Time (s)"

    for idx, bur_file in enumerate(bur_files, start=1):
        # --- main .bur ---
        df_main = _read_file(bur_file)
        if drop_last_column and df_main.shape[1] > 1:
            df_main = df_main.iloc[:, :-1]
        dfs = [df_main]

        # --- extras ---
        stem = bur_file.stem
        for ending in additional_endings:
            extra = base_path / ending / f"{stem}.{ending}"
            if not extra.exists():
                continue
            df_extra = _read_file(extra)
            # if truly empty (no cols), skip
            if df_extra.shape[1] == 0:
                continue
            if drop_last_column and df_extra.shape[1] > 1:
                df_extra = df_extra.iloc[:, :-1]
            dfs.append(df_extra)

        # horizontal concat (index‐aligned)
        combined = pd.concat(dfs, axis=1)
        # —————————————————————————————————————————————
        # drop any duplicate columns now (keep the first occurrence)
        combined = combined.loc[:, ~combined.columns.duplicated()]
        # —————————————————————————————————————————————

        # skip every Nth row
        if skip_nth_row > 1:
            combined = combined[combined.index % skip_nth_row != 0]

        # Adjust macro times if the column exists
        if macro_time_column in combined.columns and not combined.empty:
            # Convert macro times from milliseconds to seconds and add the current offset
            combined[macro_time_column] = combined[macro_time_column].astype(float) / 1000.0 + (macro_time_offset / 1000.0)

            # Rename the column to indicate it's now in seconds
            combined.rename(columns={macro_time_column: macro_time_column_seconds}, inplace=True)

            # Update the offset for the next file
            # Use the last value in the macro time column as the "max" for the next file
            # This ensures that "max is always the last" as required

            # We need to check the original DataFrames for the macro time column
            # because we've already renamed it in the combined DataFrame
            # Iterate through DataFrames in reverse order to find the last one with the macro time column
            last_value = 0
            for df in reversed(dfs):
                if macro_time_column in df.columns and not df.empty:
                    # Get the last value in milliseconds (before conversion to seconds)
                    last_value = df[macro_time_column].astype(float).iloc[-1]
                    break  # Use the first last value we find (from the last DataFrame with the column)

            # Update the offset by adding the last macro time value (still in milliseconds)
            macro_time_offset += last_value

        pieces.append(combined)

        progress.set_value(idx)
        QCoreApplication.processEvents()

    progress.set_value(len(bur_files))

    # final vertical concat
    final_df = (
        pd.concat(pieces, ignore_index=True)
        if any(len(df) for df in pieces)
        else pieces[0].iloc[0:0]  # zero‐row with correct cols if no data at all
    )

    # The macro time column has already been converted to seconds and renamed

    ds = DataSource()
    ds.data = final_df

    progress.close()
    return ds

def read_csv_sampling(filenames, sep='\t'):
    # type: (List[str])->(DataSource)
    from PyQt5.QtWidgets import QMessageBox

    if not filenames:
        return DataSource()

    # Read the first file to get parameter names and base DataFrame
    with open(filenames[0], "r") as fp:
        l = fp.readline()
        pn = l.split("\t")

    # Read the first file to get the base DataFrame
    base_df = pd.read_csv(filenames[0], sep=sep)
    row_count = len(base_df)

    # If there's only one file, process it as before
    if len(filenames) == 1:
        return DataSource(
            data=base_df,
            parameter_names=pn
        )

    # For multiple files, combine horizontally (by columns)
    combined_df = base_df

    for filename in filenames[1:]:
        df = pd.read_csv(filename, sep=sep)

        # Check if row count matches
        if len(df) != row_count:
            QMessageBox.warning(
                None, 
                "Row Count Mismatch",
                f"File {filename} has {len(df)} rows, but expected {row_count} rows. File will not be opened."
            )
            continue

        # Combine DataFrames horizontally, keeping only non-duplicate columns from the new DataFrame
        # First, identify duplicate columns
        duplicate_cols = set(combined_df.columns).intersection(set(df.columns))

        # Remove duplicate columns from the new DataFrame
        df_unique = df.drop(columns=duplicate_cols)

        # Combine with the existing DataFrame
        combined_df = pd.concat([combined_df, df_unique], axis=1)

    return DataSource(
        data=combined_df,
        parameter_names=pn
    )


def read_mfd_hdf5(filenames):
    """
    Read MFD HDF5 files, including zipped HDF5 files.
    If a .zip has no .h5/.hdf5 inside, fall back to read_burst_analysis.
    """
    from PyQt5.QtWidgets import QMessageBox

    if not filenames:
        return DataSource()

    first_file = str(filenames[0])

    # If the first input is a .zip WITHOUT any HDF5 inside, treat it as a burst ZIP.
    if first_file.lower().endswith('.zip') and not _zip_contains_any(first_file, ('.h5', '.hdf5')):
        logging.info(f"No HDF5 in zip; treating as burst analysis: {first_file}")
        return read_burst_analysis(first_file)

    # Normal HDF5 flow
    base_df = read_hdf5_file(first_file)
    row_count = len(base_df)

    if len(filenames) == 1:
        ds = DataSource()
        ds.data = base_df.select_dtypes(['number'])
        return ds

    combined_df = base_df
    for filename in filenames[1:]:
        df = read_hdf5_file(filename)
        if len(df) != row_count:
            QMessageBox.warning(
                None,
                "Row Count Mismatch",
                f"File {filename} has {len(df)} rows, but expected {row_count} rows. File will not be opened."
            )
            continue
        duplicate = set(combined_df.columns).intersection(df.columns)
        combined_df = pd.concat([combined_df, df.drop(columns=duplicate)], axis=1)

    ds = DataSource()
    ds.data = combined_df.select_dtypes(['number'])
    return ds


def read_hdf5_file(filename):
    file_path = pathlib.Path(filename)

    if file_path.suffix.lower() == '.zip':
        with zipfile.ZipFile(file_path, 'r') as zip_file:
            hdf5_files = [f for f in zip_file.namelist() if f.lower().endswith(('.h5', '.hdf5'))]
            if not hdf5_files:
                raise FileNotFoundError(f"No HDF5 files found in the zip archive: {filename}")
            hdf5_filename = hdf5_files[0]
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = pathlib.Path(temp_dir)
                zip_file.extract(hdf5_filename, temp_path)
                extracted_file = temp_path / hdf5_filename
                return pd.read_hdf(extracted_file, key='results')
    else:
        return pd.read_hdf(filename, key='results')


def read_csv(filenames):
    from PyQt5.QtWidgets import QMessageBox

    if not filenames:
        return DataSource()
    
    # Process the first file to get the base DataFrame
    first_file = filenames[0]
    base_df = read_csv_file(first_file)
    row_count = len(base_df)

    # If there's only one file, process it as before
    if len(filenames) == 1:
        dfn = base_df.select_dtypes(['number'])
        ds = DataSource()
        ds.data = dfn
        return ds

    # For multiple files, combine horizontally (by columns)
    combined_df = base_df

    for filename in filenames[1:]:
        df = read_csv_file(filename)

        # Check if row count matches
        if len(df) != row_count:
            QMessageBox.warning(
                None, 
                "Row Count Mismatch",
                f"File {filename} has {len(df)} rows, but expected {row_count} rows. File will not be opened."
            )
            continue

        # Combine DataFrames horizontally, keeping only non-duplicate columns from the new DataFrame
        # First, identify duplicate columns
        duplicate_cols = set(combined_df.columns).intersection(set(df.columns))

        # Remove duplicate columns from the new DataFrame
        df_unique = df.drop(columns=duplicate_cols)

        # Combine with the existing DataFrame
        combined_df = pd.concat([combined_df, df_unique], axis=1)

    # Select only numeric columns
    dfn = combined_df.select_dtypes(['number'])

    ds = DataSource()
    ds.data = dfn
    return ds


def read_csv_file(filename):
    """
    Read a CSV file, handling both regular and zipped CSV files.
    
    Args:
        filename: Path to the CSV file or zipped CSV file
        
    Returns:
        pandas DataFrame containing the CSV data
    """
    file_path = pathlib.Path(filename)
    
    # Check if the file is a zip file
    if file_path.suffix.lower() == '.zip':
        with zipfile.ZipFile(file_path, 'r') as zip_file:
            # Get a list of CSV files in the zip
            csv_files = [f for f in zip_file.namelist() if f.lower().endswith(('.csv', '.txt', '.dat'))]
            
            if not csv_files:
                raise ValueError(f"No CSV files found in the zip archive: {filename}")
            
            # Use the first CSV file in the archive
            csv_filename = csv_files[0]
            
            # Read the CSV file directly from the zip
            with zip_file.open(csv_filename) as csv_file:
                # Convert bytes to string for pandas
                text_io = io.TextIOWrapper(csv_file, encoding='utf-8')
                return pd.read_csv(text_io, sep="\t")
    else:
        # Regular CSV file
        return pd.read_csv(filename, sep="\t")
