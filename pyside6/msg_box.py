import sys
from PySide6.QtWidgets import (
    QApplication,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QMessageBox,
)


class MainWindow(QWidget):
    def __init__(self) -> None:
        super(MainWindow, self).__init__()
        self.setWindowTitle("Msg_Box")

        layout = QVBoxLayout()

        info_btn = QPushButton("Info msg")
        info_btn.clicked.connect(self.show_info_msg)
        layout.addWidget(info_btn)

        warning_btn = QPushButton("Warning msg")
        warning_btn.clicked.connect(self.show_warning_msg)
        layout.addWidget(warning_btn)

        question_btn = QPushButton("Qeustion msg")
        question_btn.clicked.connect(self.show_question_msg)
        layout.addWidget(question_btn)

        self.setLayout(layout)

    def show_info_msg(self):
        QMessageBox.information(
            self, "Info", "This is info msg", QMessageBox.Ok, QMessageBox.Close
        )

    def show_warning_msg(self):
        QMessageBox.information(
            self, "Warning", "This is Warning msg", QMessageBox.Ok, QMessageBox.Close
        )

    def show_question_msg(self):
        res = QMessageBox.question(
            self, "Question", "Keep going?", QMessageBox.Yes | QMessageBox.No
        )
        if res == QMessageBox.Yes:
            QMessageBox.information(
                self, "Response", "User chose 'Yes'", QMessageBox.Ok
            )
        else:
            QMessageBox.information(self, "Response", "User chose 'No'", QMessageBox.Ok)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
