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

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

try:
    from chisurf import logging
except Exception:  # pragma: no cover
    import logging  # type: ignore

from .data_source import DataSource

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QProgressBar, QLabel, QApplication, QMessageBox
from PyQt5.QtCore import Qt, QCoreApplication


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

# ----------------------------- utils -----------------------------------------

def _zip_contains_any(zip_path: str, exts: tuple[str, ...]) -> bool:
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            names = [n.lower() for n in zf.namelist()]
        return any(n.endswith(ext) for ext in exts for n in names)
    except Exception as e:
        logging.debug("Zip inspect failed for '%s': %s", zip_path, e)
        return False


class ProgressWindow(QDialog):
    def __init__(self, title="Progress", message="Processing...", max_value=100, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModal)
        layout = QVBoxLayout(self)
        self.label = QLabel(message)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, max_value)
        layout.addWidget(self.label)
        layout.addWidget(self.progress_bar)

    def set_value(self, value: int):
        self.progress_bar.setValue(value)


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
      1) Regular dirs with bi4_bur / bur (+ extras in bg4/br4/by4/bv4).
      2) Zipped MFD folders with the standard directory structure.
      3) "Selector zip" with loose .bur files (will be normalized to temp/bi4_bur).

    - Auto-detects delimiters for .bur and extras (no hardcoded tab).
    - Concatenates macro time across files → seconds; column renamed to "Mean Macro Time (s)".
    """
    # ensure a QApplication
    app = QApplication.instance() or QApplication([])

    base_path = pathlib.Path(base_path)
    additional_endings = additional_endings or ["bg4", "br4", "by4", "bv4"]

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


def _process_burst_analysis_dir(
    base_path: pathlib.Path,
    skip_nth_row: int = 2,
    additional_endings: Optional[List[str]] = None,
    drop_last_column: bool = True
) -> DataSource:
    """
    Process a burst analysis directory. Prefer HDF5 (hdf5/*.h5|*.hdf5), else read BUR files.
    """
    additional_endings = additional_endings or ["bg4", "br4", "by4", "bv4"]

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

    progress = ProgressWindow(
        title="File Processing",
        message="Processing burst files...",
        max_value=len(bur_files),
    )
    progress.show()

    pieces: List[pd.DataFrame] = []
    macro_time_offset_ms = 0.0
    macro_col_ms = "Mean Macro Time (ms)"
    macro_col_s  = "Mean Macro Time (s)"

    for i, bur in enumerate(bur_files, start=1):
        # main .bur with autodetection (fixes old sep="\t" bug)
        df_main = _read_text_table_auto(bur)
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
            df_extra = _read_text_table_auto(extra)
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

        progress.set_value(i)
        QCoreApplication.processEvents()

    progress.set_value(len(bur_files))
    progress.close()

    final_df = (
        pd.concat(pieces, ignore_index=True)
        if any(len(df) for df in pieces)
        else pieces[0].iloc[0:0]
    )

    ds = DataSource()
    ds.data = final_df
    return ds


def read_csv_sampling(filenames: List[str], sep: str = '\t') -> DataSource:
    if not filenames:
        return DataSource()

    with open(filenames[0], "r", encoding="utf-8", errors="ignore") as fp:
        pn = fp.readline().rstrip("\n").split("\t")

    base_df = pd.read_csv(filenames[0], sep=sep)
    row_count = len(base_df)

    if len(filenames) == 1:
        return DataSource(data=base_df, parameter_names=pn)

    combined_df = base_df.copy()
    for fn in filenames[1:]:
        df = pd.read_csv(fn, sep=sep)
        if len(df) != row_count:
            QMessageBox.warning(
                None,
                "Row Count Mismatch",
                f"File {fn} has {len(df)} rows, expected {row_count}. Skipping."
            )
            continue
        dup = set(combined_df.columns).intersection(df.columns)
        df_unique = df.drop(columns=list(dup)) if dup else df
        combined_df = pd.concat([combined_df, df_unique], axis=1)

    return DataSource(data=combined_df, parameter_names=pn)


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
            QMessageBox.warning(None, "Open CSV", f"Could not read file {fn}: {e}")

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
        QMessageBox.warning(
            None,
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
        with open(p, "rb") as f:
            tio = io.TextIOWrapper(f, encoding="utf-8", errors="ignore")
            head = _read_head_lines(tio)
        kwargs = _detect_and_build_kwargs(head)
        df = pd.read_csv(p, **kwargs)

    logging.info("[read_csv_file] Read %d rows from %s", len(df), filename)
    logging.info("[read_csv_file] Columns: %s", ", ".join(map(str, df.columns)))

    # normalize MSVC tokens then coerce to numeric
    df = _normalize_msvc_tokens(df)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(FILL_MISSING_VALUE)
    return df


# --------------------- low-level text-table helpers --------------------------

def _read_text_table_auto(path: pathlib.Path) -> pd.DataFrame:
    """
    Read a single text-like file (.bur/.csv/.txt/.dat) with autodetection + MSVC normalization.
    """
    with open(path, "rb") as f:
        tio = io.TextIOWrapper(f, encoding="utf-8", errors="ignore")
        head = _read_head_lines(tio)
    kwargs = _detect_and_build_kwargs(head)
    df = pd.read_csv(path, **kwargs)
    df = _normalize_msvc_tokens(df)
    # don't force numeric or fill here; leave types as read (burst pipeline often mixes ints/floats)
    # but we can still best-effort numeric:
    df = df.apply(pd.to_numeric, errors="ignore")
    return df


def _normalize_msvc_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace MSVC weird tokens with NaN/±Inf (full-cell matches).
    """
    def _norm_cell(x):
        if isinstance(x, str):
            if _WIN_NAN_RE.match(x):
                return np.nan
            if _WIN_PINF_RE.match(x):
                return np.inf
            if _WIN_NINF_RE.match(x):
                return -np.inf
        return x
    return df.applymap(_norm_cell)


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
        sep_show = repr(delim)

    if dec_comma:
        kwargs["decimal"] = ","

    # debug log (compact)
    logging.info("[detect] sep=%s, width=%s, first=%s, header=%s, data_start=%s, dec_comma=%s",
                 sep_show, complete_cols, first_idx, header_where, data_start, dec_comma)
    return kwargs
