import sys
import sip
sip.setapi('QDate', 2)
sip.setapi('QDateTime', 2)
sip.setapi('QString', 2)
sip.setapi('QTextStream', 2)
sip.setapi('QTime', 2)
sip.setapi('QUrl', 2)
sip.setapi('QVariant', 2)

from qtpy.QtWidgets import QApplication
from ndxplorer import plot_main


def main():
    app = QApplication(sys.argv)
    import numpy as np
    np.random.seed(0)
    win = plot_main.NDXplorer()
    win.show()
    app.exec_()


if __name__ == "__main__":
    main()
