import sys
import logging
from qtpy.QtWidgets import QApplication
from ndxplorer import plot_main


def main():
    # Configure logging when run as standalone module
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('ndxplorer.log')
        ]
    )
    logging.info("Starting ndxplorer as standalone module")
    
    app = QApplication(sys.argv)
    import numpy as np
    np.random.seed(0)
    win = plot_main.NDXplorer()
    win.show()
    app.exec_()


if __name__ == "__main__":
    main()
