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


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Window size change practice")
        self.resize(500, 500)

        # New btn
        self.btn = QPushButton("Click", self)
        self.btn.clicked.connect(self.buttonClicked)

        # Btn location and size setting
        self.btn.setGeometry(50, 50, 200, 50)

    def buttonClicked(self):
        print("Button has been clicked")


if __name__ == "__main__":
    # 1. Btn
    #     app = QApplication(sys.argv)
    #     win = MainWindow()
    #     win.show()
    #     sys.exit(app.exec())

    # 2. Label for text and img
    # app = QApplication([])
    # win = QWidget()

    # label = QLabel("Hi")
    # layout = QVBoxLayout()
    # layout.addWidget(label)

    # win.setLayout(layout)
    # win.resize(500, 500)
    # win.show()
    # app.exec()

    # 3. Text box
    # app = QApplication([])
    # win = QWidget()
    # line_edit = QLineEdit()
    # layout = QVBoxLayout()
    # layout.addWidget(line_edit)
    # win.setLayout(layout)
    # win.resize(500, 500)
    # win.show()
    # app.exec()

    # 4. Button
    # app = QApplication([])
    # win = QWidget()
    # btn = QPushButton("Save")
    # layout = QVBoxLayout()
    # layout.addWidget(btn)
    # win.setLayout(layout)
    # win.resize(500, 500)
    # win.show()
    # app.exec()

    # 5. CheckBox
    # app = QApplication([])
    # win = QWidget()
    # checkbox = QCheckBox("Save")
    # layout = QVBoxLayout()
    # layout.addWidget(checkbox)
    # win.setLayout(layout)
    # win.resize(500, 500)
    # win.show()
    # app.exec()

    # 6. Msg box
    app = QApplication([])
    msg_box = QMessageBox()
    msg_box.setWindowTitle("Alert")
    msg_box.setText("Done")
    msg_box.exec()
