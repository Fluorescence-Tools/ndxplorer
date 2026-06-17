# reader.py — NDXplorer Reader Module (drop-in replacement)

from __future__ import annotations
from typing import List, Union, Optional, BinaryIO, TextIO, Dict, Tuple
import pathlib
import os
import tempfile
import zipfile
import io
import shutil
import re
import json

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

try:
    from ..logging_config import logging
except Exception:  # pragma: no cover
    import logging  # type: ignore

from ..core.data_source import DataSource
from ..settings import get_settings_path, ensure_default_settings

# PyArrow support for faster CSV reading
try:
    import pyarrow as pa
    import pyarrow.csv as pa_csv
    import pyarrow.compute as pc
    HAVE_PYARROW = True
except ImportError:
    pa = None
    pa_csv = None
    pc = None
    HAVE_PYARROW = False

try:
    from qtpy.QtWidgets import QApplication, QMessageBox
    from qtpy.QtCore import QCoreApplication, QThread
    from ..ui.progress_window import ProgressWindow
    _HAS_QT = True
except Exception:
    _HAS_QT = False
    QApplication = None
    QMessageBox = None
    QCoreApplication = None
    QThread = None
    ProgressWindow = None

# Import async loading components
from .async_loader import DataLoadTask, DataLoadWorker, DataLoadResult, run_task_inline
from .file_metadata_cache import get_metadata_cache


"""
NDXplorer Reader Module (fixed)

Key improvements:
- Delimiter autodetection for text-like files (.bur/.csv/.txt/.dat) incl. zipped.
- MSVC NaN/Inf token normalization (e.g., 1.#INF, -1.#IND00e+000, 1.#QNAN).
- Robust handling of “selector zips” (loose .bur) and full MFD zips.
- Prefer HDF5 if present; fallback to BUR+extras merge.
- Concatenate macro time across files, convert to seconds, rename to "Mean Macro Time (s)".
- Deduplicate columns and select numeric for CSV/HDF5 readers.
"""

# ----------------------------- constants -------------------------------------

FILL_MISSING_VALUE = -1.0

# MSVC weird tokens as compiled regex (full-cell matches)
#  - 1.#INF, -1.#INF, 1.#IND, -1.#IND, 1.#QNAN, 1.#SNAN with optional trailing digits and exponent
_WIN_NAN_RE  = re.compile(r'^\s*[+-]?(?:\d*\.)?#(?:IND|QNAN|SNAN)\d*(?:e[+-]?\d+)?\s*$', re.IGNORECASE)
_WIN_PINF_RE = re.compile(r'^\s*\+?(?:\d*\.)?#INF\d*(?:e[+-]?\d+)?\s*$', re.IGNORECASE)
_WIN_NINF_RE = re.compile(r'^\s*-(?:\d*\.)?#INF\d*(?:e[+-]?\d+)?\s*$', re.IGNORECASE)

_TEXT_EXTS = (".bur", ".csv", ".txt", ".dat")
_HDF5_EXTS = (".h5", ".hdf5")

_DEFAULT_BURST_EXTRA_ENDINGS: List[str] = ["bg4", "br4", "by4", "bv4", "td4"]

# ----------------------------- utils -----------------------------------------

def _in_gui_thread() -> bool:
    """Return True if we're running in the main GUI thread."""
    if not _HAS_QT or QApplication is None:
        return False
    app = QApplication.instance()
    if app is None:
        return False
    try:
        return app.thread() == QThread.currentThread()
    except Exception:
        return False


def _safe_warning(title: str, message: str) -> None:
    """Show a QMessageBox when in GUI thread, otherwise fall back to logging."""
    if _HAS_QT and QApplication is not None and _in_gui_thread():
        QMessageBox.warning(None, title, message)
    else:
        logging.warning("%s: %s", title, message)

def _zip_contains_any(zip_path: str, exts: tuple[str, ...]) -> bool:
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            names = [n.lower() for n in zf.namelist()]
        return any(n.endswith(ext) for ext in exts for n in names)
    except Exception as e:
        logging.debug("Zip inspect failed for '%s': %s", zip_path, e)
        return False


# ----------------------------- PyArrow fast readers ---------------------------

def read_csv_pyarrow(
    path: Union[str, pathlib.Path],
    delimiter: str = ',',
    has_header: bool = True,
    skip_rows: int = 0,
) -> pd.DataFrame:
    """
    Read CSV file using PyArrow for maximum speed.
    
    This is 2-5x faster than pandas for large files due to:
    - Multi-threaded parsing
    - Efficient memory allocation
    - Zero-copy string handling
    
    Falls back to pandas if PyArrow fails or is unavailable.
    """
    import time
    t0 = time.perf_counter()
    
    if not HAVE_PYARROW:
        logging.info("[read_csv_pyarrow] PyArrow not available, falling back to pandas")
        return pd.read_csv(path, sep=delimiter, header=0 if has_header else None, skiprows=skip_rows)
    
    try:
        # Configure for maximum read speed
        read_options = pa_csv.ReadOptions(
            use_threads=True,
            block_size=1024 * 1024 * 32,  # 32MB blocks
            skip_rows=skip_rows,
        )
        
        parse_options = pa_csv.ParseOptions(
            delimiter=delimiter,
        )
        
        # Auto-detect column types, treating nulls appropriately
        convert_options = pa_csv.ConvertOptions(
            strings_can_be_null=True,
            null_values=['', 'NA', 'N/A', 'NaN', 'nan', 'null', 'NULL', '#N/A'],
            true_values=['true', 'True', 'TRUE', '1'],
            false_values=['false', 'False', 'FALSE', '0'],
        )
        
        if not has_header:
            read_options.autogenerate_column_names = True
        
        table = pa_csv.read_csv(
            path,
            read_options=read_options,
            parse_options=parse_options,
            convert_options=convert_options,
        )
        
        t1 = time.perf_counter()
        
        # Convert to pandas DataFrame
        df = table.to_pandas(
            self_destruct=True,  # Free Arrow memory immediately
            split_blocks=True,   # Better memory layout
            zero_copy_only=False,
        )
        
        t2 = time.perf_counter()
        logging.info("[read_csv_pyarrow] %d rows: arrow_read=%.2fs, to_pandas=%.2fs",
                     len(df), t1 - t0, t2 - t1)
        
        return df
        
    except Exception as e:
        logging.warning("[read_csv_pyarrow] PyArrow failed: %s. Falling back to pandas.", e)
        return pd.read_csv(path, sep=delimiter, header=0 if has_header else None, skiprows=skip_rows)


def read_csv_fast(
    path: Union[str, pathlib.Path],
    **kwargs,
) -> pd.DataFrame:
    """
    Read CSV with automatic engine selection for best performance.
    
    Uses PyArrow when:
    - File is large (> 10MB)
    - No complex parsing options needed
    
    Falls back to pandas C engine otherwise.
    """
    path = pathlib.Path(path)
    file_size = path.stat().st_size if path.exists() else 0
    
    # Use PyArrow for files > 10MB when available
    use_pyarrow = (
        HAVE_PYARROW
        and file_size > 10 * 1024 * 1024
        and not kwargs.get('decimal')  # PyArrow doesn't support decimal param
        and not kwargs.get('converters')
        and not kwargs.get('dtype')
    )
    
    if use_pyarrow:
        delimiter = kwargs.get('sep', kwargs.get('delimiter', ','))
        has_header = kwargs.get('header', 0) == 0
        skip_rows = kwargs.get('skiprows', 0)
        if isinstance(skip_rows, list):
            skip_rows = 0  # PyArrow doesn't support list skiprows
            use_pyarrow = False
        
        if use_pyarrow:
            try:
                return read_csv_pyarrow(path, delimiter=delimiter, has_header=has_header, skip_rows=skip_rows)
            except Exception as e:
                logging.warning("[read_csv_fast] PyArrow fallback triggered: %s", e)
    
    # Fall back to pandas
    return pd.read_csv(path, **kwargs)




def _get_burst_additional_endings() -> List[str]:
    """Return extra burst-result endings from settings, with a safe default.

    Reads ``burst_additional_endings`` from ``mfd.settings.json`` in the
    ndxplorer settings folder. Falls back to ``_DEFAULT_BURST_EXTRA_ENDINGS``
    on any error or if the key is missing/invalid.
    """

    endings: Optional[List[str]] = None

    try:
        # Ensure default settings exist and resolve the settings path
        try:
            ensure_default_settings()
        except Exception:
            pass

        settings_path = None
        try:
            settings_path = get_settings_path()
        except Exception:
            settings_path = None

        if settings_path is not None:
            cfg_path = settings_path / "mfd.settings.json"
            if cfg_path.is_file():
                with cfg_path.open("r", encoding="utf-8") as f:
                    cfg = json.load(f)
                raw = cfg.get("burst_additional_endings")
                if isinstance(raw, list):
                    cleaned: List[str] = []
                    for item in raw:
                        s = str(item).strip().lower()
                        if not s:
                            continue
                        cleaned.append(s)
                    if cleaned:
                        endings = cleaned
    except Exception as e:  # pragma: no cover - robust against config issues
        try:
            logging.debug("Failed to load burst_additional_endings from settings: %s", e)
        except Exception:
            pass

    return endings or list(_DEFAULT_BURST_EXTRA_ENDINGS)


# ----------------------------- core API --------------------------------------

def read_burst_analysis(
    base_path: Union[str, pathlib.Path],
    skip_nth_row: int = 2,
    additional_endings: Optional[List[str]] = None,
    drop_last_column: bool = True
) -> DataSource:
    """
    Read burst analysis from folder or zip.

    Supports:
      1) Regular dirs with bi4_bur / bur (+ extras in bg4/br4/by4/bv4/td4).
      2) Zipped MFD folders with the standard directory structure.
      3) "Selector zip" with loose .bur files (will be normalized to temp/bi4_bur).

    - Auto-detects delimiters for .bur and extras (no hardcoded tab).
    - Concatenates macro time across files → seconds; column renamed to "Mean Macro Time (s)".
    """
    # ensure a QApplication (skip when headless)
    if QApplication is not None:
        app = QApplication.instance() or QApplication([])

    base_path = pathlib.Path(base_path)
    if additional_endings is None:
        additional_endings = _get_burst_additional_endings()

    if base_path.is_file() and base_path.suffix.lower() == ".zip":
        # Try "selector zip" path first (loose .bur files)
        try:
            with zipfile.ZipFile(base_path, "r") as zf:
                all_files = zf.namelist()
                logging.info("Zip contains %d entries", len(all_files))
                bur_files = [f for f in all_files if f.lower().endswith(".bur")]
                if bur_files:
                    with tempfile.TemporaryDirectory() as tdir:
                        tpath = pathlib.Path(tdir)
                        bur_dir = tpath / "bi4_bur"
                        bur_dir.mkdir(parents=True, exist_ok=True)

                        # Extract .bur to /bi4_bur
                        for f in bur_files:
                            zf.extract(f, tpath)
                            src = tpath / f
                            dst = bur_dir / src.name
                            if src != dst:
                                dst.parent.mkdir(parents=True, exist_ok=True)
                                shutil.move(str(src), str(dst))

                        # Extract extras into /<ending>
                        for ending in additional_endings:
                            extra = [f for f in all_files if f.lower().endswith(f".{ending}")]
                            if not extra:
                                continue
                            ed = tpath / ending
                            ed.mkdir(parents=True, exist_ok=True)
                            for f in extra:
                                zf.extract(f, tpath)
                                src = tpath / f
                                dst = ed / src.name
                                if src != dst:
                                    shutil.move(str(src), str(dst))

                        return _process_burst_analysis_dir(
                            tpath, skip_nth_row, additional_endings, drop_last_column
                        )
        except Exception as e:
            logging.info("Direct selector-zip processing failed: %s. Falling back.", e)

        # Fallback: extract entire zip and search for MFD structure
        with tempfile.TemporaryDirectory() as tdir:
            tpath = pathlib.Path(tdir)
            with zipfile.ZipFile(base_path, "r") as zf:
                zf.extractall(tpath)

            candidates = [d for d in tpath.iterdir() if d.is_dir()]
            if len(candidates) == 1:
                mfd_dir = candidates[0]
            else:
                mfd_dir = None
                for d in candidates:
                    if (d / "hdf5").exists() or (d / "bi4_bur").exists() or (d / "bur").exists():
                        mfd_dir = d
                        break
                if mfd_dir is None:
                    mfd_dir = tpath

            return _process_burst_analysis_dir(
                mfd_dir, skip_nth_row, additional_endings, drop_last_column
            )

    # Regular dir
    return _process_burst_analysis_dir(
        base_path, skip_nth_row, additional_endings, drop_last_column
    )


def start_read_burst_analysis_async(
    base_path: Union[str, pathlib.Path],
    skip_nth_row: int = 2,
    additional_endings: Optional[List[str]] = None,
    drop_last_column: bool = True,
    on_success: Callable[["DataSource"], None] = None,
    on_error: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Start asynchronous loading of burst analysis data.
    
    In GUI mode, runs in background thread to prevent UI blocking.
    In CLI mode, runs synchronously.
    
    Parameters
    ----------
    base_path : Union[str, pathlib.Path]
        Path to burst analysis directory or zip.
    skip_nth_row : int, optional
        Skip every Nth row, by default 2.
    additional_endings : Optional[List[str]], optional
        Extra file endings to process, by default None (uses settings).
    drop_last_column : bool, optional
        Drop last column from .bur files, by default True.
    on_success : Callable[[DataSource], None], optional
        Callback when loading succeeds, by default None.
    on_error : Optional[Callable[[str], None]], optional
        Callback when loading fails, by default None.
    """
    if on_success is None:
        on_success = lambda ds: None
    if on_error is None:
        on_error = lambda msg: logging.error("Async load failed: %s", msg)
    
    task = DataLoadTask(
        description=f"Loading burst analysis from {base_path}",
        load_callable=lambda: read_burst_analysis(
            base_path, skip_nth_row, additional_endings, drop_last_column
        ),
        on_success=on_success,
        on_error=on_error,
    )
    
    if _in_gui_thread():
        worker = DataLoadWorker(task)
        worker.finished.connect(lambda result: on_success(result.data_source))
        worker.error.connect(on_error)
        worker.start()
    else:
        run_task_inline(task)


def _process_burst_analysis_dir(
    base_path: pathlib.Path,
    skip_nth_row: int = 2,
    additional_endings: Optional[List[str]] = None,
    drop_last_column: bool = True
) -> DataSource:
    """
    Process a burst analysis directory. Prefer HDF5 (hdf5/*.h5|*.hdf5), else read BUR files.
    """
    import time as _time
    t0 = _time.perf_counter()

    additional_endings = additional_endings or ["bg4", "br4", "by4", "bv4", "td4"]

    # Prefer HDF5
    hdf5_dir = base_path / "hdf5"
    if hdf5_dir.is_dir():
        h5 = sorted([p for p in hdf5_dir.iterdir() if p.suffix.lower() in _HDF5_EXTS])
        if h5:
            logging.info("Found HDF5: %s", h5[0])
            return read_mfd_hdf5([str(h5[0])])

    # BUR path
    dir_main = base_path / "bi4_bur"
    dir_fallback = base_path / "bur"
    if dir_main.is_dir():
        bur_files = sorted(dir_main.glob("*.bur")) or sorted(dir_fallback.glob("*.bur"))
    else:
        bur_files = sorted(dir_fallback.glob("*.bur"))

    if not bur_files:
        raise FileNotFoundError("No .bur files in 'bi4_bur' or 'bur'.")

    n_files = len(bur_files)
    logging.info("Processing %d .bur files from %s", n_files, base_path)

    progress = None
    if _in_gui_thread():
        progress = ProgressWindow(
            title="File Processing",
            message=f"Processing {n_files} burst files...",
            max_value=n_files,
            cancelable=True,
        )
        progress.show()

    pieces: List[pd.DataFrame] = []
    macro_time_offset_ms = 0.0
    macro_col_ms = "Mean Macro Time (ms)"
    macro_col_s  = "Mean Macro Time (s)"

    # Cache format detection from first file for speed (all .bur files share format)
    bur_format_cache: Optional[Dict] = None
    extra_format_cache: Dict[str, Dict] = {}  # ending -> kwargs

    for i, bur in enumerate(bur_files, start=1):
        # Check for cancellation
        if progress is not None and progress.was_cancelled():
            logging.info("Data loading cancelled by user")
            if progress is not None:
                progress.close()
            return DataSource()  # Return empty data source
        # Detect format from first file, reuse for rest
        if bur_format_cache is None:
            bur_format_cache = _detect_format(bur)
        df_main = _read_text_table_auto(bur, cached_kwargs=bur_format_cache)
        df_main.columns = [str(c).strip() for c in df_main.columns]
        if drop_last_column and df_main.shape[1] > 1:
            df_main = df_main.iloc[:, :-1]

        dfs = [df_main]

        # extras beside this bur (by stem) in base_path/<ending>/*.ending
        stem = bur.stem
        for ending in additional_endings:
            extra = base_path / ending / f"{stem}.{ending}"
            if not extra.exists():
                continue
            # Cache format per extra file type
            if ending not in extra_format_cache:
                extra_format_cache[ending] = _detect_format(extra)
            df_extra = _read_text_table_auto(extra, cached_kwargs=extra_format_cache[ending])
            df_extra.columns = [str(c).strip() for c in df_extra.columns]
            if df_extra.shape[1] == 0:
                continue
            if drop_last_column and df_extra.shape[1] > 1:
                df_extra = df_extra.iloc[:, :-1]
            dfs.append(df_extra)

        combined = pd.concat(dfs, axis=1)
        combined = combined.loc[:, ~combined.columns.duplicated()]

        # skip every Nth row
        if skip_nth_row > 1 and not combined.empty:
            combined = combined[combined.index % skip_nth_row != 0]

        # concatenate macro time (ms → s) and rename
        if macro_col_ms in combined.columns and not combined.empty:
            combined[macro_col_ms] = pd.to_numeric(combined[macro_col_ms], errors="coerce")
            combined[macro_col_ms] = combined[macro_col_ms] + macro_time_offset_ms
            combined.rename(columns={macro_col_ms: macro_col_s}, inplace=True)
            combined[macro_col_s] = combined[macro_col_s] / 1000.0

            # update offset (use last value from ORIGINAL df that had the macro column)
            last_ms = None
            for df in reversed(dfs):
                if macro_col_ms in df.columns and not df.empty:
                    last_ms = pd.to_numeric(df[macro_col_ms], errors="coerce").iloc[-1]
                    break
            if last_ms is None:
                last_ms = 0.0
            macro_time_offset_ms += float(last_ms)

        pieces.append(combined)

        if progress is not None:
            progress.set_value(i)
            QCoreApplication.processEvents()

    if progress is not None:
        progress.set_value(n_files)
        progress.close()

    t1 = _time.perf_counter()
    logging.info("Read %d files in %.2fs, concatenating...", n_files, t1 - t0)

    # Optimize memory usage during concatenation
    if pieces:
        # Pre-allocate list with estimated size to reduce memory reallocations
        total_rows = sum(len(df) for df in pieces if not df.empty)
        logging.info("Estimated total rows: %d", total_rows)
        
        # Use ignore_index=True and optimize dtypes before concatenation
        for i, df in enumerate(pieces):
            if not df.empty:
                # Downcast numeric columns to save memory
                pieces[i] = df.apply(pd.to_numeric, errors='coerce').convert_dtypes(convert_integer=False, convert_floating=True)
        
        # Concatenate with optimized memory settings
        final_df = pd.concat(pieces, ignore_index=True, copy=False)
    else:
        final_df = pieces[0].iloc[0:0] if pieces else pd.DataFrame()

    t2 = _time.perf_counter()
    logging.info("Burst load complete: %d rows, %.2fs total (concat %.2fs)",
                 len(final_df), t2 - t0, t2 - t1)

    ds = DataSource()
    ds.data = final_df
    return ds


def read_csv_sampling(filenames: List[str], sep: str = '\t') -> DataSource:
    """
    Read ChiSurf sampling files (.er4).
    Multiple files (chains) are concatenated by rows to form a single distribution.
    """
    if not filenames:
        return DataSource()

    dfs = []
    for fn in filenames:
        try:
            # ER4 files are tab-separated text files
            df = pd.read_csv(fn, sep=sep, header=0, comment=None)
            # Normalize column names (strip whitespace and leading #)
            df.columns = [str(c).strip().lstrip('#').strip() for c in df.columns]
            dfs.append(df)
        except Exception as e:
            logging.warning(f"Could not read sampling file {fn}: {e}")

    if not dfs:
        return DataSource()

    # Concatenate all chains by rows (stacking samples)
    combined_df = pd.concat(dfs, axis=0, ignore_index=True)
    
    # Ensure all columns are numeric (non-numeric become NaN)
    combined_df = _best_effort_numeric(combined_df)
    combined_df = _fill_missing(combined_df, FILL_MISSING_VALUE)
    
    return DataSource(data=combined_df)


def read_sampling_folder(path: str) -> DataSource:
    """
    Read a ChiSurf sampling folder. 
    It expects a 'chains/' subfolder containing .er4 files.
    """
    p = pathlib.Path(path)
    chains_dir = p / "chains"
    
    if not chains_dir.is_dir():
        # Fallback: look for .er4 files in the folder itself
        chains_dir = p
        
    er4_files = sorted(list(chains_dir.glob("*.er4")))
    if not er4_files:
        logging.warning(f"No .er4 files found in {chains_dir}")
        return DataSource()

    return read_csv_sampling([str(f) for f in er4_files])


def read_mfd_hdf5(filenames: List[str]) -> DataSource:
    """
    Read MFD HDF5 files (supports zipped HDF5).
    If a .zip has no .h5/.hdf5, fallback to read_burst_analysis(zip).
    """
    if not filenames:
        return DataSource()

    first = str(filenames[0])

    if first.lower().endswith(".zip") and not _zip_contains_any(first, _HDF5_EXTS):
        logging.info("No HDF5 in zip; treating as burst analysis: %s", first)
        return read_burst_analysis(first)

    base_df = read_hdf5_file(first)
    row_count = len(base_df)

    if len(filenames) == 1:
        ds = DataSource()
        ds.data = base_df.select_dtypes(include=["number"])
        return ds

    combined = base_df.copy()
    for fn in filenames[1:]:
        df = read_hdf5_file(fn)
        if len(df) != row_count:
            QMessageBox.warning(
                None,
                "Row Count Mismatch",
                f"File {fn} has {len(df)} rows, expected {row_count}. Skipping."
            )
            continue
        dup = set(combined.columns).intersection(df.columns)
        combined = pd.concat([combined, df.drop(columns=list(dup))], axis=1) if dup else pd.concat([combined, df], axis=1)

    ds = DataSource()
    ds.data = combined.select_dtypes(include=["number"])
    return ds


def read_hdf5_file(filename: str) -> pd.DataFrame:
    """
    Read a single HDF5 (optionally inside a .zip).
    Tries '/results' first, else the first available key.
    """
    def _read_one(h5_path: pathlib.Path) -> pd.DataFrame:
        try:
            with pd.HDFStore(str(h5_path), mode="r") as st:
                keys = st.keys()
                key = "/results" if "/results" in keys else (keys[0] if keys else "/results")
            return pd.read_hdf(h5_path, key=key)
        except Exception as e:
            # final fallback to default key
            return pd.read_hdf(h5_path)

    p = pathlib.Path(filename)
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p, "r") as zf:
            members = [f for f in zf.namelist() if f.lower().endswith(_HDF5_EXTS)]
            if not members:
                raise FileNotFoundError(f"No HDF5 files in zip: {filename}")
            target = members[0]
            with tempfile.TemporaryDirectory() as tdir:
                tpath = pathlib.Path(tdir)
                zf.extract(target, tpath)
                return _read_one(tpath / target)
    return _read_one(p)


def read_csv(filenames: List[str]) -> DataSource:
    """
    Read one or multiple CSV-like files.
    - Per-file autodetection + normalization (MSVC NaN/Inf)
    - If same ncols: stack rows; elif same nrows: stack cols (drop dups); else fallback to col-wise merge.
    """
    if not filenames:
        return DataSource()

    dfs: List[pd.DataFrame] = []
    for fn in filenames:
        try:
            df = read_csv_file(fn)
            dfs.append(df)
        except Exception as e:
            _safe_warning("Open CSV", f"Could not read file {fn}: {e}")

    if not dfs:
        return DataSource()

    if len(dfs) == 1:
        dfn = dfs[0].select_dtypes(include=["number"])
        return DataSource(data=dfn)

    ncols = [d.shape[1] for d in dfs]
    nrows = [d.shape[0] for d in dfs]

    if len(set(ncols)) == 1:
        base_cols = list(dfs[0].columns)
        norm = []
        for d in dfs:
            dd = d.copy()
            dd.columns = base_cols
            norm.append(dd)
        combined = pd.concat(norm, axis=0, ignore_index=True)

    elif len(set(nrows)) == 1:
        combined = dfs[0].copy()
        for d in dfs[1:]:
            dup = set(combined.columns).intersection(d.columns)
            d2 = d.drop(columns=list(dup)) if dup else d
            combined = pd.concat([combined, d2], axis=1)
    else:
        _safe_warning(
            "Auto-merge CSV",
            "Files share neither column count nor row count. Using column-wise merge with duplicate-column removal."
        )
        combined = dfs[0].copy()
        for d in dfs[1:]:
            dup = set(combined.columns).intersection(d.columns)
            d2 = d.drop(columns=list(dup)) if dup else d
            combined = pd.concat([combined, d2], axis=1)

    dfn = combined.select_dtypes(include=["number"])
    dfn = _fill_missing(dfn, FILL_MISSING_VALUE)
    return DataSource(data=dfn)


def start_read_csv_async(
    filenames: List[str],
    on_success: Callable[["DataSource"], None] = None,
    on_error: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Start asynchronous loading of CSV data.
    
    In GUI mode, runs in background thread to prevent UI blocking.
    In CLI mode, runs synchronously.
    
    Parameters
    ----------
    filenames : List[str]
        List of CSV file paths to read.
    on_success : Callable[[DataSource], None], optional
        Callback when loading succeeds, by default None.
    on_error : Optional[Callable[[str], None]], optional
        Callback when loading fails, by default None.
    """
    if on_success is None:
        on_success = lambda ds: None
    if on_error is None:
        on_error = lambda msg: logging.error("Async load failed: %s", msg)
    
    task = DataLoadTask(
        description=f"Loading CSV files: {filenames}",
        load_callable=lambda: read_csv(filenames),
        on_success=on_success,
        on_error=on_error,
    )
    
    if _in_gui_thread():
        worker = DataLoadWorker(task)
        worker.finished.connect(lambda result: on_success(result.data_source))
        worker.error.connect(on_error)
        worker.start()
    else:
        run_task_inline(task)


# ----------------------------- helpers ---------------------------------------

def _fill_missing(df: pd.DataFrame, sentinel: float = FILL_MISSING_VALUE) -> pd.DataFrame:
    # If you also want to neutralize ±inf: df = df.replace([np.inf, -np.inf], np.nan)
    return df.fillna(sentinel)


def coerce_numeric_majority(df: pd.DataFrame, threshold: float = 0.55, verbose: bool = False) -> pd.DataFrame:
    """
    Try to coerce mostly-numeric string columns using several locale patterns.
    """
    out = df.copy()

    def _to_numeric_series(s: pd.Series):
        s_obj = s.astype("string", copy=False)

        a = pd.to_numeric(s_obj, errors="coerce")
        a_rate = float(a.notna().mean())

        sb = s_obj.copy()
        mb = sb.notna() & sb.str.contains(",", na=False) & ~sb.str.contains(r"\.", na=False)
        if mb.any():
            sb.loc[mb] = sb.loc[mb].str.replace(",", ".", regex=False)
        b = pd.to_numeric(sb, errors="coerce")
        b_rate = float(b.notna().mean())

        sc = s_obj.copy()
        mc = sc.notna() & sc.str.contains(",", na=False)
        if mc.any():
            sc.loc[mc] = sc.loc[mc].str.replace(",", "", regex=False)
        c = pd.to_numeric(sc, errors="coerce")
        c_rate = float(c.notna().mean())

        sd = s_obj.copy()
        md = sd.notna() & sd.str.contains(r",", na=False) & sd.str.contains(r"\.", na=False)
        if md.any():
            sd.loc[md] = (sd.loc[md].str.replace(".", "", regex=False)
                                   .str.replace(",", ".", regex=False))
        d = pd.to_numeric(sd, errors="coerce")
        d_rate = float(d.notna().mean())

        candidates = [("direct", a_rate, a), ("dec_comma", b_rate, b), ("us_thousands", c_rate, c), ("eu_thousands", d_rate, d)]
        how, rate, ser = max(candidates, key=lambda x: x[1])
        return ser, rate, how

    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            if verbose:
                logging.info("[coerce_numeric_majority] %s: already numeric", col)
            continue
        ser, rate, how = _to_numeric_series(out[col])
        if rate >= threshold:
            if verbose:
                logging.info("[coerce_numeric_majority] %s → numeric (%.1f%%, %s)", col, 100*rate, how)
            out[col] = ser
        else:
            if verbose:
                logging.info("[coerce_numeric_majority] %s: keep as text (%.1f%% numeric)", col, 100*rate)
    return out


def read_csv_file(filename: str) -> pd.DataFrame:
    """
    Read a CSV-like text file or a .zip containing exactly one CSV-like text file.
    - Autodetect delimiter (, \\t ; | or whitespace).
    - Choose the first full-width *texty* row as header when available.
    - Normalize MSVC NaN/Inf/IND/QNAN/SNAN tokens.
    - Force numeric columns (non-numeric → NaN) and fill NaNs with FILL_MISSING_VALUE.
    """
    p = pathlib.Path(filename)

    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p, "r") as zf:
            inner = _find_first_member(zf, _TEXT_EXTS)
            if inner is None:
                raise ValueError(f"No CSV-like files found in zip: {filename}")
            with zf.open(inner) as bio:
                tio = io.TextIOWrapper(bio, encoding="utf-8", errors="ignore")
                head = _read_head_lines(tio)
            kwargs = _detect_and_build_kwargs(head)
            with zf.open(inner) as bio:
                tio = io.TextIOWrapper(bio, encoding="utf-8", errors="ignore")
                df = pd.read_csv(tio, **kwargs)
    else:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            head = _read_head_lines(f)
            kwargs = _detect_and_build_kwargs(head)
            f.seek(0)
            df = pd.read_csv(f, **kwargs)

    logging.info("[read_csv_file] Read %d rows from %s", len(df), filename)
    logging.info("[read_csv_file] Columns: %s", ", ".join(map(str, df.columns)))

    # normalize MSVC tokens then coerce to numeric
    df = _normalize_msvc_tokens(df)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(FILL_MISSING_VALUE)
    return df


# --------------------- low-level text-table helpers --------------------------

def _read_text_table_auto(path: pathlib.Path, cached_kwargs: Optional[Dict] = None) -> pd.DataFrame:
    """
    Read a single text-like file (.bur/.csv/.txt/.dat) with autodetection + MSVC normalization.

    Parameters
    ----------
    path : pathlib.Path
        File to read.
    cached_kwargs : Optional[Dict]
        If provided, skip detection and use these kwargs directly for pd.read_csv.
        This speeds up reading many files with the same format.

    This implementation avoids double-reading the file by rewinding the same handle after sampling.
    """
    import time as _time
    
    if cached_kwargs is not None:
        kwargs = cached_kwargs.copy()
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            head = _read_head_lines(f)
            kwargs = _detect_and_build_kwargs(head)
    
    def _load_with_kwargs(read_kwargs: Dict) -> pd.DataFrame:
        engine_local = read_kwargs.get("engine", "c")
        if engine_local == "pyarrow":
            # Use chunksize for large files to reduce memory pressure
            try:
                file_size = path.stat().st_size
                if file_size > 100 * 1024 * 1024:
                    chunks = pd.read_csv(path, chunksize=100000, **read_kwargs)
                    return pd.concat(chunks, ignore_index=True)
                return pd.read_csv(path, **read_kwargs)
            except Exception:
                # Fallback to regular read if chunked fails
                return pd.read_csv(path, **read_kwargs)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return pd.read_csv(f, **read_kwargs)

    def _drop_trailing_empty_columns(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or frame.shape[1] == 0:
            return frame
        working = frame
        while working.shape[1] > 0:
            last_col_name = working.columns[-1]
            col = working[last_col_name]
            all_nan = col.isna().all()
            all_blank = False
            if not all_nan and (pd.api.types.is_object_dtype(col) or pd.api.types.is_string_dtype(col)):
                non_null = col.dropna()
                if non_null.empty:
                    all_blank = True
                else:
                    all_blank = non_null.astype(str).str.strip().eq("").all()
            if all_nan or all_blank:
                logging.info("[read] Dropping trailing empty column '%s'", last_col_name)
                working = working.iloc[:, :-1]
                continue
            break
        return working

    def _update_cache(new_kwargs: Dict) -> None:
        if cached_kwargs is not None:
            cached_kwargs.clear()
            cached_kwargs.update(new_kwargs)

    t0 = _time.perf_counter()
    while True:
        try:
            df = _load_with_kwargs(kwargs)
            break
        except Exception as exc:
            if kwargs.get("engine") == "pyarrow":
                logging.warning("[read] PyArrow failed for %s (%s). Falling back to pandas engine.",
                                path.name, exc)
                kwargs = kwargs.copy()
                kwargs["engine"] = "c"
                kwargs.setdefault("skipinitialspace", True)
                _update_cache(kwargs)
                continue
            if cached_kwargs is not None:
                logging.warning("[read] Cached format failed for %s (%s). Re-detecting format.",
                                path.name, exc)
                kwargs = _detect_format(path)
                _update_cache(kwargs)
                t0 = _time.perf_counter()
                continue
            raise
    t1 = _time.perf_counter()

    before_drop_cols = df.shape[1]
    df = _drop_trailing_empty_columns(df)
    if df.shape[1] != before_drop_cols:
        logging.info("[read] Width reduced from %d to %d after dropping empty column(s)",
                     before_drop_cols, df.shape[1])

    engine = kwargs.get("engine", "c")
    
    # For pyarrow, convert object columns to numeric if possible - OPTIMIZED
    if engine == "pyarrow":
        object_cols_before = df.select_dtypes(include=['object']).columns
        if len(object_cols_before) > 0:
            logging.info("[read] PyArrow left %d object columns, attempting conversion", len(object_cols_before))
            
            # Pre-filter filename columns faster with vectorized operations
            filename_cols = []
            if len(object_cols_before) > 0:
                # Vectorized check for filename keywords
                col_names_lower = np.array([str(col).lower() for col in object_cols_before])
                filename_mask = np.array([any(keyword in name for keyword in ['file', 'path', 'name', 'directory']) for name in col_names_lower])
                filename_cols = list(object_cols_before[filename_mask])
                if filename_cols:
                    logging.info("[read] Skipping filename columns: %s", filename_cols)
            
            # Convert only non-filename columns in batch for better performance
            cols_to_convert = [col for col in object_cols_before if col not in filename_cols]
            if cols_to_convert:
                # Use pandas' built-in convert_dtypes for faster batch conversion
                try:
                    # First try pandas' optimized conversion
                    df[cols_to_convert] = df[cols_to_convert].convert_dtypes(convert_integer=False, convert_floating=True, convert_string=False)
                    # Then force numeric conversion for remaining object columns
                    for col in cols_to_convert:
                        if df[col].dtype == 'object':
                            df[col] = pd.to_numeric(df[col], errors='coerce', downcast='float')
                except Exception:
                    # Fallback to individual conversion
                    for col in cols_to_convert:
                        df[col] = pd.to_numeric(df[col], errors='coerce', downcast='float')
    t1_post_convert = _time.perf_counter()
    
    object_cols = df.select_dtypes(include=['object']).columns
    has_object_cols = len(object_cols) > 0
    
    t2_start = _time.perf_counter()
    if has_object_cols:
        # Skip filename columns from post-processing too
        filename_cols = [col for col in object_cols 
                       if any(keyword in col.lower() for keyword in ['file', 'path', 'name', 'directory'])]
        if filename_cols:
            logging.info("[read] Skipping filename columns in post-processing: %s", filename_cols)
        
        # Only process non-filename columns
        cols_to_process = [col for col in object_cols if col not in filename_cols]
        if cols_to_process:
            logging.info("[read] Running post-processing on %d object columns", len(cols_to_process))
            df = _normalize_msvc_tokens(df)
            df = _best_effort_numeric(df)
        else:
            logging.info("[read] Skipping post-processing - only filename columns remain")
    else:
        logging.info("[read] Skipping post-processing - no object columns")
    t2 = _time.perf_counter()
    
    logging.info("[read] %d rows, pd.read_csv=%.2fs, convert=%.2fs, post_process=%.2fs, object_cols=%d",
                 len(df), t1 - t0, t1_post_convert - t1, t2 - t2_start, len(df.select_dtypes(include=['object']).columns))
    
    return df


def _detect_format(path: pathlib.Path) -> Dict:
    """
    Detect file format and return kwargs for pd.read_csv.
    Uses cache to avoid re-detection on subsequent loads.
    """
    # Try to get cached format first
    cache = get_metadata_cache()
    cached_format = cache.get_cached_format(path)
    if cached_format is not None:
        return cached_format
    
    # Detect format and cache it
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        head = _read_head_lines(f)
        format_kwargs = _detect_and_build_kwargs(head)
    
    # Cache the detected format
    cache.cache_format(path, format_kwargs)
    return format_kwargs


def _normalize_msvc_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace MSVC weird tokens with NaN/±Inf (full-cell matches).
    Optimized: only processes columns that actually contain MSVC tokens.
    """
    if df.empty:
        return df
    
    str_cols = df.select_dtypes(include=['object']).columns
    if len(str_cols) == 0:
        return df
    
    # Quick check: sample first 100 rows to see if any MSVC tokens exist
    # This avoids expensive regex on files that don't have MSVC tokens
    sample_size = min(100, len(df))
    has_msvc = False
    for col in str_cols:
        sample = df[col].head(sample_size).astype(str)
        if sample.str.contains(r'#(?:INF|IND|QNAN|SNAN)', case=False, na=False).any():
            has_msvc = True
            break
    
    if not has_msvc:
        return df
    
    # Only copy if we actually need to modify
    result = df.copy()
    
    for col in str_cols:
        # Use vectorized replace with regex - much faster than .apply()
        series = result[col].astype(str)
        
        # Single pass replacements using pd.Series.replace with regex
        result[col] = series.replace({
            _WIN_NAN_RE: np.nan,
            _WIN_PINF_RE: np.inf,
            _WIN_NINF_RE: -np.inf,
        }, regex=True)
    
    return result


def _best_effort_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns to numeric. Fast path: just convert, don't validate.
    """
    if df.empty:
        return df

    str_cols = df.select_dtypes(include=['object']).columns
    if len(str_cols) == 0:
        return df
    
    # Fast path: convert all object columns to numeric in one go
    # Use errors='coerce' - non-numeric strings become NaN
    for col in str_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def _find_first_member(zf: zipfile.ZipFile, exts: Tuple[str, ...]) -> Optional[str]:
    for name in zf.namelist():
        if name.lower().endswith(exts):
            return name
    return None


def _read_head_lines(file_like: TextIO, max_lines: int = 5000) -> List[str]:
    lines: List[str] = []
    for _ in range(max_lines):
        ln = file_like.readline()
        if not ln:
            break
        if isinstance(ln, bytes):
            try:
                ln = ln.decode("utf-8", errors="ignore")
            except Exception:
                ln = ln.decode("latin-1", errors="ignore")
        lines.append(str(ln))
    return lines


def _tokenize(s: str, delim: Optional[str]) -> List[str]:
    s = s.rstrip("\n")
    if not s.strip():
        return []
    return s.strip().split() if delim is None else s.strip().split(delim)


def _has_alpha(tokens: List[str]) -> bool:
    return any(any(c.isalpha() for c in t) for t in tokens)


def _detect_and_build_kwargs(lines: List[str]) -> Dict:
    """
    Detect delimiter and header row from a sample of lines.
    Returns kwargs for pandas.read_csv.
    """
    N = min(len(lines), 5000)
    candidates = [",", "\t", ";"]
    has_comma_digits = any(re.search(r"\d,\d", ln) for ln in lines[:N])

    best = dict(score=-1.0, delim=None, complete_cols=0, first_idx=0, header_prev_idx=None, dec_comma=False)

    for delim in candidates:
        ncols: List[int] = []
        for i in range(N):
            toks = _tokenize(lines[i], delim)
            ncols.append(len(toks) if len(toks) >= 2 else 0)

        counts: Dict[int, int] = {}
        for c in ncols:
            if c >= 2:
                counts[c] = counts.get(c, 0) + 1
        if not counts:
            continue

        # pick most frequent width; tie → larger width
        complete_cols = max(sorted(counts.keys()), key=lambda c: (counts[c], c))
        try:
            first_complete = next(i for i, c in enumerate(ncols) if c == complete_cols)
        except StopIteration:
            continue

        header_prev_idx = None
        if first_complete > 0:
            prev = first_complete - 1
            prev_cols = ncols[prev]
            enough = max(2, complete_cols // 2)
            if prev_cols == 0:
                header_prev_idx = prev
            elif 2 <= prev_cols < complete_cols and prev_cols >= enough:
                header_prev_idx = prev
            if header_prev_idx is None and prev_cols == 0 and prev - 1 >= 0 and ncols[prev - 1] >= enough:
                header_prev_idx = prev - 1

        # score: run length of consistent rows * log(width)
        run_len = 0
        j = first_complete
        while j < N and ncols[j] == complete_cols:
            run_len += 1
            j += 1
        score = run_len * float(np.log(max(complete_cols, 2)))
        dec_flag = bool(has_comma_digits and (delim in (";", "\t", "|", None)))

        if score > best["score"]:
            best.update(score=score, delim=delim, complete_cols=complete_cols,
                        first_idx=first_complete, header_prev_idx=header_prev_idx, dec_comma=dec_flag)

    first_idx = best["first_idx"]
    header_prev_idx = best["header_prev_idx"]
    delim = best["delim"]
    complete_cols = best["complete_cols"]
    dec_comma = best["dec_comma"]

    # pick header: prefer first full-width *texty* row
    use_header_idx = None
    first_toks = _tokenize(lines[first_idx], delim)
    if len(first_toks) == complete_cols and _has_alpha(first_toks):
        use_header_idx = first_idx
    elif header_prev_idx is not None:
        prev_toks = _tokenize(lines[header_prev_idx], delim)
        if len(prev_toks) == complete_cols and _has_alpha(prev_toks):
            use_header_idx = header_prev_idx

    kwargs: Dict = dict(skipinitialspace=True)
    if use_header_idx is not None:
        kwargs["header"] = 0
        kwargs["skiprows"] = use_header_idx
        data_start = use_header_idx + 1
        header_where = f"line {use_header_idx}"
    else:
        kwargs["header"] = None
        kwargs["skiprows"] = first_idx
        data_start = first_idx
        header_where = "none"

    if delim is None:
        kwargs["delim_whitespace"] = True
        kwargs["engine"] = "python"
        sep_show = "<whitespace>"
    else:
        kwargs["sep"] = delim
        # Use pyarrow engine for speed if available and compatible
        # pyarrow doesn't support: decimal comma, skiprows, skipinitialspace
        can_use_pyarrow = (
            not dec_comma
            and first_idx == 0
            and use_header_idx in (None, 0)
        )
        if can_use_pyarrow:
            try:
                import pyarrow  # noqa: F401
                kwargs["engine"] = "pyarrow"
                kwargs.pop("skipinitialspace", None)  # not supported by pyarrow
            except ImportError:
                kwargs["engine"] = "c"
        else:
            kwargs["engine"] = "c"
        sep_show = repr(delim)

    if dec_comma:
        kwargs["decimal"] = ","

    # debug log (compact) - only shown once per file type due to caching
    logging.info("[detect] sep=%s, width=%d, engine=%s, header=%s, data_start=%d",
                 sep_show, complete_cols, kwargs.get("engine", "c"), header_where, data_start)
    return kwargs
