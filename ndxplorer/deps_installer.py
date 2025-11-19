"""
Self-contained dependency installer utilities for NDXplorer.

Provides helpers to install optional features (napari, umap-learn, hdbscan, ...)
into the current ChiSurf environment without depending on ChiSurf's updater.

Key functionality:
- Discover conda executable within the active environment or PATH.
- Build and execute conda install commands with optional Windows elevation.
- Fallback to pip installation when requested.
- Qt-based user prompts with clear risk warnings.
- Post-install verification via `conda list` and/or import testing.

Notes:
- Installing extra packages may break the ChiSurf environment; inform the user.
- On Windows, when ChiSurf is installed under Program Files, elevation may be
  required; this module can run an elevated batch file and persist logs.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

from .logging_config import logging

# Qt imports (runtime optional)
try:
    from chisurf.gui import QtWidgets, QtCore, QtGui
except Exception:
    try:
        from qtpy import QtWidgets, QtCore, QtGui  # type: ignore
    except Exception:
        QtWidgets = None  # type: ignore
        QtCore = None  # type: ignore
        QtGui = None  # type: ignore


# ------------------------------
# Low-level command execution
# ------------------------------

def _run_command(cmd: List[str]) -> Tuple[bool, Optional[str], str, str]:
    """
    Run a command, capture stdout/stderr.

    Returns: (ok, error_message, stdout, stderr)
    """
    try:
        logging.debug(f"deps_installer: executing: {' '.join(cmd)}")
        popen_cmd = cmd
        use_shell = False
        if os.name == 'nt':
            exe = (cmd[0] if cmd else '').lower()
            if exe.endswith('.bat') or exe.endswith('.cmd'):
                popen_cmd = ['cmd.exe', '/C', *cmd]
                logging.debug("deps_installer: wrapping batch with cmd.exe /C on Windows")
        proc = subprocess.Popen(
            popen_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=use_shell,
        )
        out, err = proc.communicate()
        if out:
            logging.debug(f"deps_installer stdout:\n{out}")
        if err:
            logging.debug(f"deps_installer stderr:\n{err}")
        if proc.returncode != 0:
            return False, f"Exit code {proc.returncode}", out, err
        return True, None, out, err
    except Exception as e:
        return False, str(e), '', ''


def _needs_elevation(env_prefix: str) -> bool:
    """
    Heuristic to decide if we may need elevation to modify the environment.
    On Windows, if env is under Program Files or not writable.
    On POSIX, if env is under /usr (not /usr/local) or not writable.
    """
    try:
        env_path = Path(env_prefix)
        if os.name == 'nt':
            program_files = os.environ.get('ProgramFiles', r'C:\\Program Files')
            program_files_x86 = os.environ.get('ProgramFiles(x86)', r'C:\\Program Files (x86)')
            if str(env_path).startswith(program_files) or str(env_path).startswith(program_files_x86):
                return True
        else:
            if str(env_path).startswith('/usr') and not str(env_path).startswith('/usr/local'):
                return True
        # Try a write test inside env
        test_dir = env_path / 'conda-meta'
        if not test_dir.exists():
            test_dir = env_path
        test_file = test_dir / ('.write_test_' + next(tempfile._get_candidate_names()))
        try:
            with open(test_file, 'w') as f:
                f.write('ok')
            test_file.unlink(missing_ok=True)  # type: ignore[arg-type]
            return False
        except Exception:
            return True
    except Exception:
        # Be conservative
        return False


def _run_with_elevation(cmd: List[str]) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Windows-only: run command elevated, logging output to a persistent file.
    Returns: (ok, error_message, log_file_path)
    On non-Windows, falls back to normal run.
    """
    if os.name != 'nt':
        ok, err, _, _ = _run_command(cmd)
        return ok, err, None
    try:
        temp_dir = tempfile.mkdtemp(prefix='ndxplorer_elev_')
        log_file = os.path.join(temp_dir, 'elevated_command.log')
        # Properly quote args with spaces
        quoted = [f'"{a}"' if (' ' in str(a) and not str(a).startswith('"')) else str(a) for a in cmd]
        win_cmd = ' '.join(quoted)
        batch_file = os.path.join(temp_dir, 'run_elevated.bat')
        with open(batch_file, 'w', encoding='utf-8') as f:
            f.write('@echo off\n')
            f.write(f'echo Running elevated command at %DATE% %TIME% > "{log_file}"\n')
            f.write(f'echo Command: {win_cmd} >> "{log_file}"\n')
            f.write(f'{win_cmd} >> "{log_file}" 2>&1\n')
            f.write('set EXITCODE=%ERRORLEVEL%\n')
            f.write(f'echo. >> "{log_file}"\n')
            f.write(f'echo Exit code: %EXITCODE% >> "{log_file}"\n')
            f.write('if %EXITCODE% NEQ 0 (\n')
            f.write(f'  echo Elevated command failed with error code %EXITCODE% >> "{log_file}"\n')
            f.write('  exit /b %EXITCODE%\n')
            f.write(')\n')
            f.write('echo Elevated command completed successfully >> "{log_file}"\n')
            f.write('exit /b 0\n')
        ps = [
            'powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command',
            f"$p = Start-Process -FilePath '{batch_file}' -Verb RunAs -Wait -PassThru; exit $p.ExitCode",
        ]
        logging.debug(f"deps_installer: running elevated batch: {batch_file}")
        logging.debug(f"deps_installer: elevated log: {log_file}")
        proc = subprocess.Popen(ps, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = proc.communicate()
        if out:
            logging.debug(f"elevation launcher stdout:\n{out}")
        if err:
            logging.debug(f"elevation launcher stderr:\n{err}")
        if proc.returncode != 0:
            # Provide tail hint
            tail_hint = ''
            try:
                if os.path.exists(log_file):
                    with open(log_file, 'r', errors='ignore') as lf:
                        lines = lf.readlines()
                        tail_hint = ''.join(lines[-25:]).strip()
            except Exception:
                pass
            msg = f"Elevation failed with exit code {proc.returncode}. See log: {log_file}"
            if tail_hint:
                msg += f"\n--- Log tail ---\n{tail_hint}"
            return False, msg, log_file
        return True, None, log_file
    except Exception as e:
        return False, str(e), None


# ------------------------------
# Conda and pip helpers
# ------------------------------

def find_conda_executable() -> str:
    """Try to locate a conda executable suitable for running commands."""
    conda_exe = os.environ.get('CONDA_EXE', '')
    if conda_exe and os.path.exists(conda_exe):
        return conda_exe
    candidates: list[str] = []
    if os.name == 'nt':
        candidates.extend([
            str(Path(sys.prefix) / 'Scripts' / 'conda.exe'),
            str(Path(sys.prefix) / 'condabin' / 'conda.bat'),
            str(Path(os.environ.get('CONDA_PREFIX', '')) / 'Scripts' / 'conda.exe'),
            str(Path(os.environ.get('CONDA_PREFIX', '')) / 'condabin' / 'conda.bat'),
        ])
    else:
        candidates.extend([
            str(Path(sys.prefix) / 'bin' / 'conda'),
            str(Path(os.environ.get('CONDA_PREFIX', '')) / 'bin' / 'conda'),
        ])
    for c in candidates:
        if c and os.path.exists(c):
            os.environ['CONDA_EXE'] = c
            return c
    return 'conda'  # hope it's on PATH


def find_mamba_executable() -> Optional[str]:
    """Locate mamba or micromamba if available, preferring mamba.

    Returns an absolute path when possible. If not found, returns None.
    """
    # Check env hints first
    for var in ('MAMBA_EXE', 'MAMBA', 'MICROMAMBA_EXE'):
        p = os.environ.get(var, '')
        if p and os.path.exists(p):
            return p
    candidates: list[str] = []
    if os.name == 'nt':
        # Common Windows paths inside the current env or CONDA_PREFIX
        candidates.extend([
            str(Path(sys.prefix) / 'Scripts' / 'mamba.exe'),
            str(Path(sys.prefix) / 'condabin' / 'mamba.bat'),
            str(Path(sys.prefix) / 'Scripts' / 'micromamba.exe'),
            str(Path(os.environ.get('CONDA_PREFIX', '')) / 'Scripts' / 'mamba.exe'),
            str(Path(os.environ.get('CONDA_PREFIX', '')) / 'condabin' / 'mamba.bat'),
            str(Path(os.environ.get('CONDA_PREFIX', '')) / 'Scripts' / 'micromamba.exe'),
        ])
    else:
        candidates.extend([
            str(Path(sys.prefix) / 'bin' / 'mamba'),
            str(Path(sys.prefix) / 'bin' / 'micromamba'),
            str(Path(os.environ.get('CONDA_PREFIX', '')) / 'bin' / 'mamba'),
            str(Path(os.environ.get('CONDA_PREFIX', '')) / 'bin' / 'micromamba'),
        ])
    for c in candidates:
        if c and os.path.exists(c):
            return c
    # Fallback to PATH resolution using shutil.which
    try:
        from shutil import which
    except Exception:
        which = None  # type: ignore
    if which is not None:
        path = which('mamba')
        if path:
            # Prefer native exe over BAT wrappers when possible
            p_lower = path.lower()
            if p_lower.endswith('mamba.bat') or p_lower.endswith('mamba.cmd'):
                try:
                    # Try to reconstruct base prefix and use Scripts\mamba.exe
                    base = Path(path).parents[2]  # .../Library/Bin -> base
                    exe_candidate = base / 'Scripts' / 'mamba.exe'
                    if exe_candidate.exists():
                        return str(exe_candidate)
                except Exception:
                    pass
            return path
        path = which('micromamba')
        if path:
            return path
    return None


def choose_solver() -> Tuple[str, str]:
    """
    Choose the best available conda-family solver executable and its name.
    Returns (exe_path_or_name, solver_name) where solver_name in { 'mamba', 'micromamba', 'conda' }.
    """
    m = find_mamba_executable()
    if m:
        # Try to infer the solver name from path/basename
        base = os.path.basename(m).lower()
        solver_name = 'micromamba' if 'micro' in base else 'mamba'
        logging.debug(f"deps_installer: choosing solver {solver_name} at: {m}")
        return m, solver_name
    c = find_conda_executable()
    logging.debug(f"deps_installer: choosing solver conda at: {c}")
    return c, 'conda'


def conda_install(packages: List[str], channels: Optional[List[str]] = None, update_deps: bool = True) -> Tuple[bool, Optional[str]]:
    """
    Install given packages into current env (sys.prefix) via conda-family solver.
    Prefers mamba/micromamba when available, falls back to conda.
    """
    channels = channels or ['conda-forge', 'defaults']
    solver_exe, solver_name = choose_solver()
    env_path = sys.prefix
    cmd: List[str] = [
        solver_exe, 'install', '-y', '--prefix', env_path,
    ]
    if update_deps and solver_name == 'conda':
        # Only conda needs --update-deps explicitly; mamba/micromamba update deps by default
        cmd.append('--update-deps')
    cmd.extend(packages)
    for ch in channels:
        cmd.extend(['-c', ch])
    logging.info(f"{solver_name} install -> env: {env_path}")
    logging.debug("solver command: " + ' '.join(cmd))
    if _needs_elevation(env_path):
        ok, err, _ = _run_with_elevation(cmd)
        if not ok:
            return False, err
    else:
        ok, err, _, _ = _run_command(cmd)
        if not ok:
            return False, err
    # Diagnostics: list packages via conda (or solver)
    try:
        list_cmd = [solver_exe, 'list', '--prefix', env_path, *packages]
        logging.debug("list command: " + ' '.join(list_cmd))
        _ok, _err, out, _ = _run_command(list_cmd)
        if not _ok:
            logging.warning(f"list command reported an issue: {_err}")
        else:
            logging.debug(f"list output:\n{out}")
    except Exception as e:
        logging.debug(f"list diagnostic failed: {e}")
    return True, None


def pip_install(packages: List[str]) -> Tuple[bool, Optional[str]]:
    """Install packages via pip into the current environment."""
    cmd = [sys.executable, '-m', 'pip', 'install', '--upgrade', *packages]
    logging.info("pip install -> packages: " + ', '.join(packages))
    logging.debug("pip command: " + ' '.join(cmd))
    ok, err, _, _ = _run_command(cmd)
    return ok, err


def try_import(import_name: str) -> Tuple[bool, Optional[str]]:
    try:
        mod = __import__(import_name)
        ver = getattr(mod, '__version__', 'unknown')
        logging.info(f"Import test succeeded for {import_name}, version: {ver}")
        return True, None
    except Exception as e:
        logging.warning(f"Import test failed for {import_name}: {e}")
        return False, str(e)


# ------------------------------
# User-facing ensure/install
# ------------------------------

RISK_NOTE = (
    "Warning: Installing additional packages can change the ChiSurf environment. "
    "This may break your ChiSurf installation and could require reinstalling ChiSurf."
)


class InstallProgressDialog(QtWidgets.QDialog):  # type: ignore[misc]
    """Simple dialog that streams install output into a text area."""
    def __init__(self, parent=None, title="Installing"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(900, 560)
        layout = QtWidgets.QVBoxLayout(self)

        # Status and options row
        top_row = QtWidgets.QHBoxLayout()
        self.statusLabel = QtWidgets.QLabel("Running...")
        self.statusLabel.setStyleSheet("color: #555;")
        self.chkAutoClose = QtWidgets.QCheckBox("Close this window automatically on success")
        self.chkAutoClose.setChecked(True)
        top_row.addWidget(self.statusLabel)
        top_row.addStretch(1)
        top_row.addWidget(self.chkAutoClose)
        layout.addLayout(top_row)

        self.text = QtWidgets.QTextEdit(self)
        self.text.setReadOnly(True)
        self.text.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        layout.addWidget(self.text)

        btns = QtWidgets.QHBoxLayout()
        self.btnCopy = QtWidgets.QPushButton("Copy All")
        self.btnSave = QtWidgets.QPushButton("Save Log As…")
        self.btnOpen = QtWidgets.QPushButton("Open Log Folder")
        self.btnCancel = QtWidgets.QPushButton("Cancel")
        btns.addWidget(self.btnCopy)
        btns.addWidget(self.btnSave)
        btns.addWidget(self.btnOpen)
        btns.addStretch(1)
        btns.addWidget(self.btnCancel)
        layout.addLayout(btns)

        self.process: Optional[QtCore.QProcess] = None  # type: ignore[name-defined]
        self._log_path: Optional[str] = None
        self._timer: Optional[QtCore.QTimer] = None  # type: ignore[name-defined]
        self._finished = False
        self.btnCopy.clicked.connect(self._copy)
        self.btnSave.clicked.connect(self._save)
        self.btnOpen.clicked.connect(self._open)
        self.btnCancel.clicked.connect(self._cancel)

    def append(self, s: str):
        self.text.moveCursor(QtGui.QTextCursor.End)
        self.text.insertPlainText(s)
        self.text.moveCursor(QtGui.QTextCursor.End)
        QtWidgets.QApplication.processEvents()

    def set_log_path(self, p: Optional[str]):
        self._log_path = p

    def set_finished_state(self, success: bool, extra_msg: Optional[str] = None):
        self._finished = True
        if success:
            self.statusLabel.setText("Finished successfully")
            self.statusLabel.setStyleSheet("color: #2d7;")
        else:
            self.statusLabel.setText("Finished with errors")
            self.statusLabel.setStyleSheet("color: #d33;")
            if extra_msg:
                self.append("\n" + extra_msg + "\n")
                self.append("You can Copy All or Save the log, or open the log folder if elevation was used.\n")
        self.btnCancel.setText("Close")
        self.btnCancel.setEnabled(True)

    def wait_for_user_close(self):
        # Block until dialog is closed by user
        self.setWindowModality(QtCore.Qt.ApplicationModal)  # type: ignore[attr-defined]
        self.exec_()

    def _copy(self):
        QtWidgets.QApplication.clipboard().setText(self.text.toPlainText())

    def _save(self):
        try:
            default_name = "install_log.txt"
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Log As", default_name, "Text Files (*.txt);;All Files (*.*)")
            if path:
                with open(path, 'w', encoding='utf-8', errors='ignore') as f:
                    f.write(self.text.toPlainText())
        except Exception:
            pass

    def _open(self):
        if not self._log_path:
            return
        folder = os.path.dirname(self._log_path)
        try:
            if os.name == 'nt':
                os.startfile(folder)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(['xdg-open', folder])
        except Exception:
            pass

    def _cancel(self):
        if not self._finished:
            if self.process is not None and self.process.state() != QtCore.QProcess.NotRunning:  # type: ignore[attr-defined]
                self.process.kill()
            self.btnCancel.setEnabled(False)
        else:
            self.close()

    # Non-elevated path: use QProcess to stream stdout/err
    def run_streaming(self, cmd: List[str]) -> Tuple[bool, Optional[str]]:
        self.append("Command: " + ' '.join(cmd) + "\n\n")
        self.process = QtCore.QProcess(self)  # type: ignore[name-defined]
        # Ensure UTF-8
        env = QtCore.QProcessEnvironment.systemEnvironment()  # type: ignore[name-defined]
        env.insert("PYTHONUTF8", "1")
        # Help avoid buggy/slow third-party conda plugins
        env.insert("CONDA_NO_PLUGINS", "true")
        self.process.setProcessEnvironment(env)
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)  # type: ignore[name-defined]
        self.process.readyReadStandardOutput.connect(lambda: self.append(str(self.process.readAllStandardOutput(), 'utf-8', 'ignore')))  # type: ignore
        self.process.readyReadStandardError.connect(lambda: self.append(str(self.process.readAllStandardError(), 'utf-8', 'ignore')))  # type: ignore
        # Windows: wrap bat/cmd via cmd.exe /C
        popen_cmd = cmd
        if os.name == 'nt':
            exe = (cmd[0] if cmd else '').lower()
            if exe.endswith('.bat') or exe.endswith('.cmd'):
                popen_cmd = ['cmd.exe', '/C', *cmd]
        # Pre-flight: if the executable looks like a path, ensure it exists
        exe_path = popen_cmd[0]
        if os.path.sep in exe_path or (os.name == 'nt' and '/' in exe_path):
            if not os.path.exists(exe_path):
                msg = f"Installer executable not found: {exe_path}"
                self.append(msg + "\n")
                return False, msg
        self.process.start(popen_cmd[0], popen_cmd[1:])
        if not self.process.waitForStarted(5000):
            try:
                err_str = self.process.errorString()
            except Exception:
                err_str = ""
            detail = f"Failed to start installer process. {err_str}".strip()
            detail += "\nCommand: " + ' '.join(cmd)
            return False, detail
        # Modal loop until finished
        while True:
            if self.process.waitForFinished(100):
                break
            QtWidgets.QApplication.processEvents()
        code = self.process.exitCode()
        ok = (code == 0)
        return ok, (None if ok else f"Exit code {code}")

    # Elevated path: tail a logfile until 'Exit code:' appears
    def tail_log_until_exitcode(self, log_file: str) -> Tuple[bool, Optional[str]]:
        self.set_log_path(log_file)
        self.append(f"Tailing elevated log: {log_file}\n\n")
        last_size = 0
        exit_code: Optional[int] = None
        while True:
            try:
                if os.path.exists(log_file):
                    with open(log_file, 'r', errors='ignore') as lf:
                        lf.seek(last_size)
                        chunk = lf.read()
                        if chunk:
                            self.append(chunk)
                            last_size += len(chunk.encode('utf-8', 'ignore'))
                        # Look for exit code line
                        for line in chunk.splitlines():
                            if line.strip().lower().startswith('exit code:'):
                                try:
                                    exit_code = int(line.split(':', 1)[1].strip())
                                except Exception:
                                    exit_code = 1
                                break
                if exit_code is not None:
                    break
            except Exception:
                pass
            QtWidgets.QApplication.processEvents()
            QtCore.QThread.msleep(150)  # type: ignore[name-defined]
        ok = (exit_code == 0)
        return ok, (None if ok else f"Elevated process exit code {exit_code}")


def _prepare_elevated_batch(cmd: List[str]) -> Tuple[bool, Optional[str], Optional[str]]:
    """Create elevated batch + log and start it (Windows). Returns (ok, err, log_path)."""
    if os.name != 'nt':
        return False, "Not Windows", None
    try:
        temp_dir = tempfile.mkdtemp(prefix='ndxplorer_elev_')
        log_file = os.path.join(temp_dir, 'elevated_command.log')
        quoted = [f'"{a}"' if (' ' in str(a) and not str(a).startswith('"')) else str(a) for a in cmd]
        win_cmd = ' '.join(quoted)
        batch_file = os.path.join(temp_dir, 'run_elevated.bat')
        with open(batch_file, 'w', encoding='utf-8') as f:
            f.write('@echo off\n')
            f.write('chcp 65001 > nul\n')
            f.write(f'echo Running elevated command at %DATE% %TIME% > "{log_file}"\n')
            f.write(f'echo Command: {win_cmd} >> "{log_file}"\n')
            f.write(f'{win_cmd} >> "{log_file}" 2>&1\n')
            f.write('set EXITCODE=%ERRORLEVEL%\n')
            f.write(f'echo. >> "{log_file}"\n')
            f.write(f'echo Exit code: %EXITCODE% >> "{log_file}"\n')
            f.write('exit /b %EXITCODE%\n')
        ps = [
            'powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command',
            f"Start-Process -FilePath '{batch_file}' -Verb RunAs -WindowStyle Hidden"
        ]
        # Fire and forget; no wait, we will tail the log
        subprocess.Popen(ps)
        logging.debug(f"Started elevated batch: {batch_file}; log: {log_file}")
        return True, None, log_file
    except Exception as e:
        return False, str(e), None


def install_with_progress(parent, packages: List[str], channels: Optional[List[str]] = None, use_pip: bool = False) -> Tuple[bool, Optional[str]]:
    """Run install with a modal progress dialog streaming output or tailing log.

    Behavior:
    - While running: dialog streams output and is modal.
    - On success: auto-close if the checkbox is enabled; otherwise keep open until user closes.
    - On failure: never auto-close; require user to click Close so the error is visible.
    """
    if QtWidgets is None:
        # Fallback to non-GUI execution
        if use_pip:
            return pip_install(packages)
        return conda_install(packages, channels=channels)

    dlg = InstallProgressDialog(parent, title=f"Installing {' '.join(packages)}")

    # Helper to run a command and, if it fails, keep the dialog open for next attempts
    def _attempt(cmd_list: List[str]) -> Tuple[bool, Optional[str]]:
        dlg.append("=== Attempt ===\n")
        return dlg.run_streaming(cmd_list)

    # Detect elevated need once for conda-family (pip never elevates)
    channels = channels or ['conda-forge', 'defaults']
    env_path = sys.prefix

    if use_pip:
        # Special-case napari on Windows: prefer extras to pull GUI backends
        pkgs = packages[:]
        if os.name == 'nt' and len(pkgs) == 1 and pkgs[0].lower() == 'napari':
            pkgs = ['napari[all]']
        cmd = [sys.executable, '-m', 'pip', 'install', '--upgrade', *pkgs]
        need_elev = False
        solver_name = 'pip'
        solver_exe = sys.executable
    else:
        solver_exe, solver_name = choose_solver()
        base_cmd = [solver_exe, 'install', '-y', '--prefix', env_path]
        # Only conda requires --update-deps explicitly; mamba/micromamba do it by default
        if solver_name == 'conda':
            base_cmd.append('--update-deps')
        base_cmd.extend(packages)
        for ch in channels:
            base_cmd.extend(['-c', ch])
        cmd = base_cmd
        need_elev = _needs_elevation(env_path)

    ok = False
    err: Optional[str] = None

    # Optional preflight: search availability to provide early feedback (non-fatal)
    try:
        if not use_pip:
            search_cmd = [solver_exe, 'search', *packages]
            for ch in channels:
                search_cmd.extend(['-c', ch])
            dlg.show()
            dlg.append("Preflight: searching for packages...\n")
            _ok_search, _ = dlg.run_streaming(search_cmd)
            if not _ok_search:
                dlg.append("Preflight search did not succeed; proceeding with install anyway.\n\n")
    except Exception:
        pass

    if not need_elev:
        # Attempt 1: chosen solver, as composed
        dlg.show()
        ok, err = _attempt(cmd)
        if not ok and not use_pip:
            # Attempt 2: relax channel priority and prefer only conda-forge
            dlg.append("\n=== Retry with no channel priority and only conda-forge ===\n")
            retry_cmd = [solver_exe, 'install', '-y', '--prefix', env_path]
            if solver_name == 'conda':
                retry_cmd.append('--update-deps')
            retry_cmd.extend(packages)
            retry_cmd.append('--no-channel-priority')
            retry_cmd.extend(['-c', 'conda-forge'])
            ok, err = _attempt(retry_cmd)

        if not ok and not use_pip:
            # Attempt 3: conda fallback with plugins disabled
            dlg.append("\n=== Retry with conda --no-plugins and no channel priority ===\n")
            conda_exe = find_conda_executable()
            retry_cmd2 = [conda_exe, 'install', '-y', '--prefix', env_path, '--update-deps', '--no-channel-priority']
            retry_cmd2.extend(packages)
            for ch in channels:
                retry_cmd2.extend(['-c', ch])
            ok, err = _attempt(retry_cmd2)

        if not ok:
            # Attempt 4: pip fallback (with napari[all] on Windows)
            dlg.append("\n=== Fallback to pip ===\n")
            pkgs = packages[:]
            if os.name == 'nt' and len(pkgs) == 1 and pkgs[0].lower() == 'napari':
                pkgs = ['napari[all]']
            pip_cmd = [sys.executable, '-m', 'pip', 'install', '--upgrade', *pkgs]
            ok, err = _attempt(pip_cmd)

        # Mark finished and decide whether to keep the window
        dlg.set_finished_state(ok, err)
        if ok and dlg.chkAutoClose.isChecked():
            dlg.close()
        else:
            dlg.wait_for_user_close()
            dlg.close()
    else:
        # Elevated: for now run a single elevated attempt; advise manual retry if needed
        ok_prep, err_prep, log_path = _prepare_elevated_batch(cmd)
        if not ok_prep or not log_path:
            # Show the error in the dialog so it doesn't disappear
            dlg.show()
            dlg.set_log_path(None)
            dlg.append((err_prep or 'Failed to start elevated process') + "\n")
            dlg.set_finished_state(False, err_prep)
            dlg.wait_for_user_close()
            dlg.close()
            return False, err_prep or 'Failed to start elevated process'
        dlg.set_log_path(log_path)
        dlg.show()
        # Tail until exit code appears
        ok, err = dlg.tail_log_until_exitcode(log_file=log_path)
        if not ok:
            dlg.append("\nTip: You can retry with 'no channel priority' or fall back to pip from a terminal if elevation is required.\n")
        dlg.set_finished_state(ok, err)
        if ok and dlg.chkAutoClose.isChecked():
            dlg.close()
        else:
            dlg.wait_for_user_close()
            dlg.close()

    # Post-install verification (best-effort)
    try:
        if ok and not use_pip:
            solver_exe_v, _solver_name_v = choose_solver()
            list_cmd = [solver_exe_v, 'list', '--prefix', env_path, *packages]
            dlg = None  # avoid reusing closed dialog in non-modal context
            _run_command(list_cmd)
    except Exception:
        pass

    return ok, err


def handle_import_error(package_name: str, parent=None) -> None:
    """
    General handler for ImportError exceptions on conda-installable packages.
    Shows a dialog directing the user to ChiSurf's Package Manager.
    
    This can be called from except ImportError blocks throughout ChiSurf code
    to provide consistent user guidance for missing optional packages.
    
    Args:
        package_name: Name of the package that failed to import
        parent: Parent widget for the dialog (optional)
    """
    if QtWidgets is None or QtCore is None:
        logging.warning(f"Package '{package_name}' not available. Please use ChiSurf's Package Manager to install it.")
        return
    
    title = f"Package {package_name} not available"
    text = (
        f"The package '{package_name}' is required but not installed.\n\n"
        f"Please use ChiSurf's Package Manager (available in Help > Updates and Packages > Package Manager) "
        f"to install the '{package_name}' package."
    )
    
    msg = QtWidgets.QMessageBox(parent)
    msg.setIcon(QtWidgets.QMessageBox.Information)
    msg.setWindowTitle(title)
    msg.setText(text)
    msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
    msg.exec_()
