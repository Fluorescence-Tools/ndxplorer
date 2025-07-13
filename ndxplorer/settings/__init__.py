"""
Settings management for ndxplorer.

This module provides functions for managing ndxplorer settings, including
determining the settings path and copying default settings to the user's
settings folder.
"""

import os
import pathlib
import shutil
import importlib.util

def get_settings_path() -> pathlib.Path:
    """
    Get the path to the ndxplorer settings folder.
    
    If chisurf is installed, this will be a subfolder of the user's chisurf
    settings folder. Otherwise, it will be the default settings folder in the
    ndxplorer module directory.
    
    Returns:
        pathlib.Path: Path to the ndxplorer settings folder
    """
    # Check if chisurf is installed
    chisurf_spec = importlib.util.find_spec("chisurf")
    
    if chisurf_spec is not None:
        # chisurf is installed, use its settings path
        try:
            from chisurf.settings.path_utils import get_path
            chisurf_settings_path = get_path('settings')
            settings_path = chisurf_settings_path / 'ndxplorer'
        except ImportError:
            # Fall back to default path if there's an error importing chisurf
            settings_path = pathlib.Path(__file__).parent
    else:
        # chisurf is not installed, use default path
        settings_path = pathlib.Path(__file__).parent
    
    # Create the settings directory if it doesn't exist
    settings_path.mkdir(parents=True, exist_ok=True)
    
    return settings_path

def ensure_default_settings():
    """
    Ensure that default settings files exist in the user's settings folder.
    
    If the settings folder doesn't exist or is empty, copy the default settings
    from the ndxplorer module directory.
    """
    settings_path = get_settings_path()
    default_settings_path = pathlib.Path(__file__).parent
    
    # If settings_path is the same as default_settings_path, no need to copy
    if settings_path == default_settings_path:
        return
    
    # Check if settings files exist in the user's settings folder
    settings_files = list(settings_path.glob('*.json')) + list(settings_path.glob('*.yaml'))
    
    # If no settings files exist, copy the default settings
    if not settings_files:
        for file in default_settings_path.iterdir():
            if file.is_file() and (file.suffix == '.json' or file.suffix == '.yaml'):
                shutil.copy2(file, settings_path / file.name)
