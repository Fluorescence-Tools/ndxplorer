"""
Performance Settings Dialog for ndxplorer.

Provides a user-friendly interface for configuring performance-related
environment variables and settings with detailed tooltips.
"""

from typing import Dict, Any
from qtpy import QtWidgets, QtCore
import json
from pathlib import Path

from ..logging_config import logging
from ..utils.performance_config import (
    get_performance_config, 
    set_performance_config,
    reset_performance_config,
    PerformanceConfig
)


class PerformanceSettingsDialog(QtWidgets.QDialog):
    """Dialog for configuring performance settings with tooltips."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Performance Settings")
        self.setModal(True)
        self.resize(600, 700)
        
        # Store original config for reset functionality
        self._original_config = get_performance_config()
        
        self._setup_ui()
        self._load_current_settings()
        
    def _setup_ui(self):
        """Setup the user interface."""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Create scroll area for settings
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_widget)
        
        # Add title
        title = QtWidgets.QLabel("Performance Configuration")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        scroll_layout.addWidget(title)
        
        # Add description
        desc = QtWidgets.QLabel(
            "Configure performance settings for ndxplorer. These settings control histogram computation, "
            "memory usage, and optimization features. Changes take effect after restarting ndxplorer."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("margin: 0px 10px 20px 10px; color: #666;")
        scroll_layout.addWidget(desc)
        
        # Create settings groups
        self._create_background_group(scroll_layout)
        self._create_computation_group(scroll_layout)
        self._create_memory_group(scroll_layout)
        self._create_optimization_group(scroll_layout)
        
        # Add stretch
        scroll_layout.addStretch()
        
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        # Add buttons
        button_layout = QtWidgets.QHBoxLayout()
        
        self.reset_btn = QtWidgets.QPushButton("Reset to Defaults")
        self.reset_btn.clicked.connect(self._reset_to_defaults)
        self.reset_btn.setToolTip("Reset all performance settings to their default values")
        
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.apply_btn.clicked.connect(self._apply_settings)
        
        self.ok_btn = QtWidgets.QPushButton("OK")
        self.ok_btn.clicked.connect(self._ok_clicked)
        self.ok_btn.setDefault(True)
        
        button_layout.addWidget(self.reset_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.ok_btn)
        
        layout.addLayout(button_layout)
        
    def _create_background_group(self, layout):
        """Create background computation settings group."""
        group = QtWidgets.QGroupBox("Background Computation")
        group.setToolTip("Configure background thread processing for responsive UI")
        group_layout = QtWidgets.QVBoxLayout(group)
        
        # Background worker checkbox
        self.bg_worker_cb = QtWidgets.QCheckBox("Enable Background Worker")
        self.bg_worker_cb.setToolTip(
            "When enabled, histogram computations run in a background thread for responsive UI.\n"
            "When disabled, computations run immediately but may block the UI during processing.\n"
            "Disable on very low-memory systems or if experiencing threading issues."
        )
        group_layout.addWidget(self.bg_worker_cb)
        
        layout.addWidget(group)
        
    def _create_computation_group(self, layout):
        """Create histogram computation settings group."""
        group = QtWidgets.QGroupBox("Histogram Computation")
        group.setToolTip("Configure histogram computation engine and threading")
        group_layout = QtWidgets.QVBoxLayout(group)
        
        # Boost histogram checkbox
        self.boost_hist_cb = QtWidgets.QCheckBox("Use Boost-Histogram")
        self.boost_hist_cb.setToolTip(
            "boost-histogram provides significantly faster histogram computation with better\n"
            "memory efficiency than numpy. Falls back to numpy if unavailable.\n"
            "Disable if boost-histogram causes compatibility issues."
        )
        group_layout.addWidget(self.boost_hist_cb)
        
        # Numba checkbox
        self.numba_cb = QtWidgets.QCheckBox("Use Numba JIT Compilation")
        self.numba_cb.setToolTip(
            "Numba compiles Python code to machine code for significant speedup in\n"
            "numerical operations. May cause slight startup delays for first compilation.\n"
            "Disable if Numba causes startup delays or compatibility issues."
        )
        group_layout.addWidget(self.numba_cb)
        
        # Fast histogram checkbox
        self.fast_hist_cb = QtWidgets.QCheckBox("Use Fast Histogram Optimizations")
        self.fast_hist_cb.setToolTip(
            "Enables various histogram optimization techniques beyond boost-histogram.\n"
            "Includes caching strategies and optimized algorithms.\n"
            "Keep enabled for best performance."
        )
        group_layout.addWidget(self.fast_hist_cb)
        
        # Parallel histogram checkbox
        self.parallel_cb = QtWidgets.QCheckBox("Use Parallel Computation")
        self.parallel_cb.setToolTip(
            "Use multiple CPU cores for histogram computation when available.\n"
            "Provides significant speedup for large datasets.\n"
            "Disable on single-core systems or if threading causes issues."
        )
        group_layout.addWidget(self.parallel_cb)
        
        # Threads setting
        threads_layout = QtWidgets.QHBoxLayout()
        threads_label = QtWidgets.QLabel("Histogram Threads:")
        threads_label.setToolTip(
            "Number of threads to use for histogram computation.\n"
            "-1: Auto-detect and use all available cores\n"
            "0 or 1: Disable parallelism\n"
            "2+: Use specified number of threads"
        )
        self.threads_spin = QtWidgets.QSpinBox()
        self.threads_spin.setRange(-1, 64)
        self.threads_spin.setValue(-1)
        self.threads_spin.setToolTip(
            "Number of threads for histogram computation.\n"
            "-1 uses all available cores, 0/1 disables parallelism."
        )
        threads_layout.addWidget(threads_label)
        threads_layout.addWidget(self.threads_spin)
        threads_layout.addStretch()
        group_layout.addLayout(threads_layout)
        
        # Plot backend selection
        backend_layout = QtWidgets.QHBoxLayout()
        backend_label = QtWidgets.QLabel("Plot Backend:")
        backend_label.setToolTip(
            "Choose the plotting backend for 2D plots and visualizations:\n"
            "• guiqwt: Fast, lightweight (default, recommended)\n"
            "• matplotlib: Feature-rich, widely used\n"
            "• pyqtgraph: Fast interactive plots\n"
            "Changes require restarting ndxplorer to take effect."
        )
        self.backend_combo = QtWidgets.QComboBox()
        self.backend_combo.addItems(["guiqwt", "matplotlib", "pyqtgraph"])
        self.backend_combo.setToolTip(
            "Select plotting backend. guiqwt is recommended for best performance."
        )
        backend_layout.addWidget(backend_label)
        backend_layout.addWidget(self.backend_combo)
        backend_layout.addStretch()
        group_layout.addLayout(backend_layout)
        
        layout.addWidget(group)
        
    def _create_memory_group(self, layout):
        """Create memory and caching settings group."""
        group = QtWidgets.QGroupBox("Memory & Caching")
        group.setToolTip("Configure memory usage and caching strategies")
        group_layout = QtWidgets.QVBoxLayout(group)
        
        # Histogram cache checkbox
        self.hist_cache_cb = QtWidgets.QCheckBox("Enable Histogram Cache")
        self.hist_cache_cb.setToolTip(
            "Cache computed histograms to avoid recomputation during UI interactions.\n"
            "Improves responsiveness for repeated operations but uses more memory.\n"
            "Disable if memory is extremely limited."
        )
        group_layout.addWidget(self.hist_cache_cb)
        
        # Bitfield masks checkbox
        self.bitfield_cb = QtWidgets.QCheckBox("Use Bitfield Masks")
        self.bitfield_cb.setToolTip(
            "Use compact bitfield masks for data selection, reducing memory usage by ~8x\n"
            "for large datasets. Provides significant memory savings for selections.\n"
            "Disable if experiencing mask-related issues."
        )
        group_layout.addWidget(self.bitfield_cb)
        
        # Aggressive caching checkbox
        self.aggressive_cb = QtWidgets.QCheckBox("Aggressive Caching")
        self.aggressive_cb.setToolTip(
            "Cache more intermediate results for better performance at the cost of\n"
            "higher memory usage. Improves responsiveness for complex operations.\n"
            "Disable on memory-constrained systems."
        )
        group_layout.addWidget(self.aggressive_cb)
        
        # Memory limits
        memory_layout = QtWidgets.QGridLayout()
        
        # Histogram cache memory
        hist_cache_label = QtWidgets.QLabel("Histogram Cache (MB):")
        hist_cache_label.setToolTip(
            "Maximum memory usage for histogram cache in megabytes.\n"
            "Larger values improve performance but use more RAM.\n"
            "Recommended range: 50-2000 MB"
        )
        self.hist_cache_spin = QtWidgets.QSpinBox()
        self.hist_cache_spin.setRange(10, 4000)
        self.hist_cache_spin.setValue(200)
        self.hist_cache_spin.setToolTip("Histogram cache memory limit in MB")
        memory_layout.addWidget(hist_cache_label, 0, 0)
        memory_layout.addWidget(self.hist_cache_spin, 0, 1)
        
        # General cache memory
        gen_cache_label = QtWidgets.QLabel("General Cache (MB):")
        gen_cache_label.setToolTip(
            "Memory limit for general-purpose caching (selections, computations, etc.).\n"
            "Affects various UI operations and data processing.\n"
            "Recommended range: 10-500 MB"
        )
        self.gen_cache_spin = QtWidgets.QSpinBox()
        self.gen_cache_spin.setRange(5, 1000)
        self.gen_cache_spin.setValue(50)
        self.gen_cache_spin.setToolTip("General cache memory limit in MB")
        memory_layout.addWidget(gen_cache_label, 1, 0)
        memory_layout.addWidget(self.gen_cache_spin, 1, 1)
        
        # Bitfield threshold
        threshold_label = QtWidgets.QLabel("Bitfield Threshold:")
        threshold_label.setToolTip(
            "Minimum number of data points required to use bitfield masks.\n"
            "Below this threshold, regular boolean masks are used.\n"
            "Recommended range: 10,000-1,000,000 points"
        )
        self.threshold_spin = QtWidgets.QSpinBox()
        self.threshold_spin.setRange(1000, 10000000)
        self.threshold_spin.setValue(100000)
        self.threshold_spin.setToolTip("Minimum points for bitfield masks")
        memory_layout.addWidget(threshold_label, 2, 0)
        memory_layout.addWidget(self.threshold_spin, 2, 1)
        
        group_layout.addLayout(memory_layout)
        layout.addWidget(group)
        
    def _create_optimization_group(self, layout):
        """Create additional optimization settings group."""
        group = QtWidgets.QGroupBox("Advanced Options")
        group.setToolTip("Advanced performance optimization options")
        group_layout = QtWidgets.QVBoxLayout(group)
        
        # Info label
        info = QtWidgets.QLabel(
            "For most users, the default settings provide optimal performance. "
            "Advanced users can fine-tune these settings for specific workloads."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #666; font-style: italic; margin: 5px;")
        group_layout.addWidget(info)
        
        # Settings file info
        settings_info = QtWidgets.QLabel(
            "Settings are saved to: ~/.chisurf/ndxplorer/mfd.settings.json\n"
            "Values in settings file override environment variables."
        )
        settings_info.setWordWrap(True)
        settings_info.setStyleSheet("color: #666; font-size: 11px; margin: 5px;")
        group_layout.addWidget(settings_info)
        
        layout.addWidget(group)
        
    def _load_current_settings(self):
        """Load current settings into the UI."""
        config = get_performance_config()
        
        # Get background worker setting
        try:
            from ..plotting.plot_control import is_background_computation_enabled
            bg_enabled = is_background_computation_enabled()
        except Exception:
            bg_enabled = True  # Default fallback
        
        # Background settings
        self.bg_worker_cb.setChecked(bg_enabled)
        
        # Computation settings
        self.boost_hist_cb.setChecked(config.use_boost_histogram)
        self.numba_cb.setChecked(config.use_numba)
        self.fast_hist_cb.setChecked(config.use_fast_histogram)
        self.parallel_cb.setChecked(config.parallel_histogram)
        self.threads_spin.setValue(config.histogram_threads)
        
        # Plot backend
        backend_index = self.backend_combo.findText(config.plot_backend)
        if backend_index >= 0:
            self.backend_combo.setCurrentIndex(backend_index)
        else:
            logging.warning(f"Unknown plot backend '{config.plot_backend}', using default")
        
        # Memory settings
        self.hist_cache_cb.setChecked(config.use_histogram_cache)
        self.bitfield_cb.setChecked(config.use_bitfield_masks)
        self.aggressive_cb.setChecked(config.aggressive_caching)
        self.hist_cache_spin.setValue(int(config.histogram_cache_memory_mb))
        self.gen_cache_spin.setValue(int(config.general_cache_memory_mb))
        self.threshold_spin.setValue(config.bitfield_threshold)
        
    def _apply_settings(self):
        """Apply settings and save to settings file."""
        try:
            # Create new config
            new_config = PerformanceConfig(
                use_bitfield_masks=self.bitfield_cb.isChecked(),
                use_histogram_cache=self.hist_cache_cb.isChecked(),
                use_boost_histogram=self.boost_hist_cb.isChecked(),
                use_numba=self.numba_cb.isChecked(),
                use_fast_histogram=self.fast_hist_cb.isChecked(),
                histogram_cache_memory_mb=float(self.hist_cache_spin.value()),
                general_cache_memory_mb=float(self.gen_cache_spin.value()),
                bitfield_threshold=self.threshold_spin.value(),
                parallel_histogram=self.parallel_cb.isChecked(),
                aggressive_caching=self.aggressive_cb.isChecked(),
                histogram_threads=self.threads_spin.value(),
            )
            
            # Save to settings file
            self._save_to_settings_file(new_config)
            
            # Apply globally
            set_performance_config(new_config)
            
            QtWidgets.QMessageBox.information(
                self, 
                "Settings Applied", 
                "Performance settings have been saved.\n\n"
                "Some changes may require restarting ndxplorer to take full effect."
            )
            
            logging.info("Performance settings updated and saved")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, 
                "Error", 
                f"Failed to save settings: {e}"
            )
            logging.error(f"Failed to save performance settings: {e}")
            
    def _save_to_settings_file(self, config: PerformanceConfig):
        """Save configuration to settings file."""
        try:
            # Get settings path
            from ..settings import get_settings_path
            settings_path = get_settings_path()
            settings_file = settings_path / "mfd.settings.json"
            
            # Load existing settings
            if settings_file.exists():
                with open(settings_file, 'r', encoding='utf-8') as f:
                    settings_data = json.load(f)
            else:
                settings_data = {}
            
            # Update environment section
            settings_data['environment'] = {
                'NDXPLORER_ENABLE_BACKGROUND_WORKER': self.bg_worker_cb.isChecked(),
                'NDXPLORER_USE_BOOST_HISTOGRAM': config.use_boost_histogram,
                'NDXPLORER_USE_NUMBA': config.use_numba,
                'NDXPLORER_USE_FAST_HISTOGRAM': config.use_fast_histogram,
                'NDXPLORER_USE_HISTOGRAM_CACHE': config.use_histogram_cache,
                'NDXPLORER_PARALLEL_HISTOGRAM': config.parallel_histogram,
                'NDXPLORER_HISTOGRAM_THREADS': config.histogram_threads,
                'NDXPLORER_USE_BITFIELD': config.use_bitfield_masks,
                'NDXPLORER_BITFIELD_THRESHOLD': config.bitfield_threshold,
                'NDXPLORER_AGGRESSIVE_CACHING': config.aggressive_caching,
                'NDXPLORER_HISTOGRAM_CACHE_MB': config.histogram_cache_memory_mb,
                'NDXPLORER_GENERAL_CACHE_MB': config.general_cache_memory_mb,
            }
            
            # Save settings
            settings_file.parent.mkdir(parents=True, exist_ok=True)
            with open(settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings_data, f, indent=2, sort_keys=True)
                
        except Exception as e:
            raise RuntimeError(f"Failed to save settings file: {e}")
            
    def _reset_to_defaults(self):
        """Reset all settings to defaults."""
        reply = QtWidgets.QMessageBox.question(
            self, 
            "Reset Settings", 
            "Reset all performance settings to their default values?\n\n"
            "This will clear your custom configuration.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            reset_performance_config()
            self._load_current_settings()
            QtWidgets.QMessageBox.information(self, "Reset", "Settings reset to defaults")
            
    def _ok_clicked(self):
        """Handle OK button click."""
        self._apply_settings()
        self.accept()
