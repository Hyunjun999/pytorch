import sys
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QLabel,
    QVBoxLayout,
    QWidget,
    QLineEdit,
    QCheckBox,
    QMessageBox,
)


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Basic widget practice")
        self.label = QLabel("Enter the locatio info")
        self.line_edit = QLineEdit()
        self.checkbox = QCheckBox("Agreement for passing location info")
        self.send_button = QPushButton("Sending")
        layer = QVBoxLayout()
        layer.addWidget(self.label)
        layer.addWidget(self.line_edit)
        layer.addWidget(self.checkbox)
        layer.addWidget(self.send_button)

        self.setLayout(layer)
        self.send_button.clicked.connect(self.show_msg)

    def show_msg(self):
        if self.checkbox.isChecked():
            msg = self.line_edit.text()
            print(f"Input : {msg}")
            self.line_edit.clear()
            self.checkbox.setChecked(False)
        else:
            error = "Button has not been clicked"
            QMessageBox.critical(self, "Error", error)
            self.line_edit.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
