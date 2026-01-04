from qtpy.QtWidgets import QDoubleSpinBox, QApplication, QWidget, QVBoxLayout, QLabel
from qtpy.QtCore import Qt
from qtpy.QtGui import QValidator
import re
import sys


class ScientificSpinBox(QDoubleSpinBox):
    def __init__(self, parent=None, format_str="%.5e", suffix="", use_si=False, relative_step=0.1, **kwargs):
        # Custom attributes (must come before super().__init__)
        self.format_str = format_str
        self._suffix = suffix
        self.use_si = use_si
        self.relative_step = relative_step  # e.g. 0.1 = 10%
        self.si_prefixes = {
            -12: "p", -9: "n", -6: "µ", -3: "m", 0: "",
             3: "k", 6: "M", 9: "G", 12: "T"
        }

        super().__init__(parent, **kwargs)
        decimals = kwargs.get("decimals", 15)
        self.setDecimals(decimals)
        self.setAlignment(Qt.AlignRight)
        self.setKeyboardTracking(False)

    def textFromValue(self, value):
        if self.use_si:
            return self._format_si(value)
        else:
            return self.format_str % value + (" " + self._suffix if self._suffix else "")

    def valueFromText(self, text):
        try:
            txt = text.replace(self._suffix, "").strip()
            value = self._parse_si(txt) if self.use_si else float(txt)
            return value
        except Exception:
            return self.value()

    def validate(self, text, pos):
        pattern = re.compile(r"^\s*[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?\s*([a-zA-Zµ]*)?\s*$")
        if pattern.match(text.strip()):
            return QValidator.Acceptable, text, pos
        else:
            return QValidator.Intermediate, text, pos

    def _format_si(self, value):
        if value == 0:
            return f"0 {self._suffix}".strip()
        exp = int("{:e}".format(value).split("e")[1])
        exp3 = 3 * int(exp // 3)
        exp3 = max(min(exp3, 12), -12)
        scaled = value / (10 ** exp3)
        prefix = self.si_prefixes.get(exp3, f"e{exp3}")
        decimals = self.decimals() if hasattr(self, 'decimals') else 3
        return f"{scaled:.{decimals}f} {prefix}{self._suffix}".strip()

    def _parse_si(self, text):
        text = text.strip()
        for exp, prefix in self.si_prefixes.items():
            if prefix and text.endswith(prefix):
                try:
                    num = float(text[:-len(prefix)].strip())
                    return num * (10 ** exp)
                except ValueError:
                    continue
        return float(text)

    def stepBy(self, steps: int):
        current = self.value()
        delta = self.relative_step * abs(current)
        if delta == 0:
            delta = self.relative_step  # default if current == 0
        new_value = current + steps * delta
        self.setValue(min(max(new_value, self.minimum()), self.maximum()))


# Demo
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = QWidget()
    layout = QVBoxLayout(win)

    label = QLabel("Value:")
    spinbox = ScientificSpinBox(format_str="%.3e", suffix="mol/L", use_si=True)
    spinbox.setRange(1e-15, 1e3)
    spinbox.setValue(1.23e-6)

    def update_label(val):
        label.setText(f"Value: {val:.3e}")

    spinbox.valueChanged.connect(update_label)

    layout.addWidget(spinbox)
    layout.addWidget(label)
    win.show()
    sys.exit(app.exec_())