"""
Settings management for ndxplorer.

This module provides functions for managing ndxplorer settings, including
determining the settings path and copying default settings to the user's
settings folder.
"""

import os
import pathlib
import shutil

def get_settings_path() -> pathlib.Path:
    """
    Get the path to the ndxplorer settings folder.
    
    Uses a user-local settings directory at ~/.ndxplorer/, falling back to
    the module directory if that cannot be created.
    
    Returns:
        pathlib.Path: Path to the ndxplorer settings folder
    """
    try:
        settings_path = pathlib.Path.home() / ".ndxplorer"
        settings_path.mkdir(parents=True, exist_ok=True)
    except Exception:
        settings_path = pathlib.Path(__file__).parent
    
    return settings_path

def ensure_default_settings():
    """
    Ensure that default settings files exist in the user's settings folder.
    
    If the settings folder doesn't exist or is empty, copy the default settings
    from the ndxplorer module directory, including the 'templates' subfolder.
    """
    try:
        settings_path = get_settings_path()
        default_settings_path = pathlib.Path(__file__).parent
        
        # If settings_path is the same as default_settings_path, no need to copy
        if settings_path == default_settings_path:
            return
        
        # Check if settings files exist in the user's settings folder
        settings_files = list(settings_path.glob('*.json')) + list(settings_path.glob('*.yaml'))
        
        # If no settings files exist, copy the default settings files in root
        if not settings_files:
            import logging
            logging.info(f"Copying default settings to {settings_path}")
            for file in default_settings_path.iterdir():
                if file.is_file() and (file.suffix == '.json' or file.suffix == '.yaml'):
                    shutil.copy2(file, settings_path / file.name)
        
        # Ensure templates directory exists and copy defaults if missing
        src_templates = default_settings_path / 'templates'
        dst_templates = settings_path / 'templates'
        if src_templates.exists() and src_templates.is_dir():
            if not dst_templates.exists():
                shutil.copytree(src_templates, dst_templates)
    except Exception as e:
        import logging
        logging.error(f"Failed to ensure default settings: {e}")
