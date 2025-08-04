import sys
from qtpy.QtWidgets import QApplication
from ndxplorer import plot_main
from ndxplorer.logging_config import logging


def main():
    logging.info("Starting ndxplorer as standalone module")
    
    app = QApplication(sys.argv)
    import numpy as np
    np.random.seed(0)
    win = plot_main.NDXplorer()
    win.show()
    app.exec_()


if __name__ == "__main__":
    main()
