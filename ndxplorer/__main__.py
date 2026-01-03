import sys
import os
import argparse
from pathlib import Path

# Add the ndxplorer module to Python path for direct execution
if __name__ == "__main__" and __package__ is None:
    # Get the parent directory of this file (ndxplorer package root)
    current_dir = Path(__file__).parent
    module_root = current_dir.parent.parent  # Go up to chisurf root
    
    # Add both the ndxplorer module and chisurf root to path
    sys.path.insert(0, str(module_root))
    sys.path.insert(0, str(current_dir.parent))
    
    # Set the package name
    __package__ = "ndxplorer"

from qtpy.QtWidgets import QApplication
from .core.plot_main import NDXplorer
from .logging_config import logging
from pathlib import Path


def open_path_like_drop(ndxplorer, path_str):
    """
    Open a file or folder using the same logic as file drops.
    
    This mimics the behavior of the dropEvent in working_path_helpers.py
    """
    from pathlib import Path
    
    path = Path(path_str)
    
    if not path.exists():
        logging.error(f"Path does not exist: {path}")
        return
    
    if path.is_dir():
        # Handle directory like burst_dir drop
        logging.info(f"Opening as directory: {path}")
        try:
            ndxplorer.lineEditWorkingPath.setText(str(path))
        except Exception:
            pass
        try:
            ndxplorer.open_files(file_type="burst_dir", file_handles=str(path), append=False)
        except Exception as exc:
            logging.error(f"Failed to open directory: {exc}")
    
    elif path.is_file():
        # Handle file based on extension like file drops
        suffix = path.suffix.lower()
        logging.info(f"Opening as file ({suffix}): {path}")
        
        try:
            if suffix == ".csv":
                ndxplorer.onOpenCsv(None, filenames=[str(path)], append=False, merge_mode="columns")
            elif suffix in (".h5", ".hdf5"):
                ndxplorer.onOpenMfdHdf5(None, filenames=[str(path)], append=False, merge_mode="columns")
            elif suffix in (".bur", ".txt"):
                # Try opening as burst files
                ndxplorer.open_files(file_type="burst", file_handles=str(path), append=False)
            else:
                # Default behavior for other file types
                logging.info(f"Unknown file type {suffix}, trying default open")
                ndxplorer.open_files(str(path))
        except Exception as exc:
            logging.error(f"Failed to open file: {exc}")
    
    else:
        logging.error(f"Path is neither file nor directory: {path}")


def parse_arguments():
    """Parse command line arguments for NDxplorer."""
    parser = argparse.ArgumentParser(
        description='NDXplorer - Fluorescence Data Explorer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Open with specific folder
  python ndxplorer/__main__.py --folder "E:\\eGFP_bad_background\\pxl_eGFP_bad_background"
  
  # Open with specific file
  python ndxplorer/__main__.py --file "data.bur"
  
  # Use default test data
  python ndxplorer/__main__.py --test-data
  
  # Open empty application
  python ndxplorer/__main__.py
        """
    )
    
    # File/folder options
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--file', '-f',
        type=str,
        help='Open specific file (.bur, .csv, etc.)'
    )
    group.add_argument(
        '--folder', '-d',
        type=str,
        help='Open folder containing data files'
    )
    group.add_argument(
        '--test-data', '-t',
        action='store_true',
        help='Open with default test data path (E:\\eGFP_bad_background\\pxl_eGFP_bad_background)'
    )
    
    # Additional options
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for NDxplorer CLI."""
    args = parse_arguments()
    
    # Set up logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Debug logging enabled")
    
    logging.info("Starting ndxplorer as standalone module")
    
    # Create Qt application
    app = QApplication(sys.argv)
    import numpy as np
    np.random.seed(0)
    
    # Create main window
    win = NDXplorer()
    win.show()
    
    # Handle file/folder arguments using the same logic as file drops
    if args.test_data:
        # Try multiple possible test data paths
        test_paths = [
            r"E:\eGFP_bad_background\pxl_eGFP_bad_background",
            r"Q:\tttr-data\imaging\zeiss\eGFP_bad_background\pxl_eGFP_bad_background",
            r"E:\dev\chisurf\test\data\pxl_eGFP_bad_background",
            r"E:\dev\chisurf\modules\ndxplorer\test\data"
        ]
        
        test_path = None
        for path in test_paths:
            if os.path.exists(path):
                test_path = path
                break
        
        if test_path:
            logging.info(f"Opening test data: {test_path}")
            open_path_like_drop(win, test_path)
        else:
            logging.warning("No test data path found. Available options tried:")
            for path in test_paths:
                logging.warning(f"  - {path}")
            logging.info("Opening empty application instead")
    elif args.file:
        logging.info(f"Opening file: {args.file}")
        if os.path.exists(args.file):
            open_path_like_drop(win, args.file)
        else:
            logging.error(f"File not found: {args.file}")
    elif args.folder:
        logging.info(f"Opening folder: {args.folder}")
        if os.path.exists(args.folder):
            open_path_like_drop(win, args.folder)
        else:
            logging.error(f"Folder not found: {args.folder}")
    else:
        logging.info("Opening empty application")
    
    # Start the event loop
    app.exec_()


if __name__ == "__main__":
    main()
