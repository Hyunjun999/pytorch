import sys
import csv
from typing import Optional
import PySide6.QtCore
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QDialog,
    QMessageBox,
    QListWidget,
)


class InputWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Type some info")

        self.age_line_edit = QLineEdit()
        self.gender_line_edit = QLineEdit()
        self.country_line_edit = QLineEdit()

        # Button
        self.view_btn = QPushButton("view")

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(QLabel("age : "))
        layout.addWidget(self.age_line_edit)

        layout.addWidget(QLabel("gender : "))
        layout.addWidget(self.gender_line_edit)

        layout.addWidget(QLabel("country : "))
        layout.addWidget(self.country_line_edit)

        layout.addWidget(self.view_btn)

        self.setLayout(layout)  # Upload full layout to the display
        self.view_btn.clicked.connect(self.show_info)

    def show_info(self):
        age = self.age_line_edit.text()
        gender = self.gender_line_edit.text()
        country = self.country_line_edit.text()

        info_win = InfoWindow(age, gender, country)
        info_win.setModal(True)
        info_win.exec()


class InfoWindow(QDialog):
    def __init__(self, age, gender, country) -> None:
        super().__init__()
        self.setWindowTitle("Info confirm")

        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"age : {age}"))
        layout.addWidget(QLabel(f"gender : {gender}"))
        layout.addWidget(QLabel(f"country : {country}"))

        save_btn = QPushButton("Save")
        close_btn = QPushButton("Close")
        load_btn = QPushButton("Load")

        layout.addWidget(save_btn)
        layout.addWidget(close_btn)
        layout.addWidget(load_btn)

        self.setLayout(layout)

        save_btn.clicked.connect(lambda: self.save_info(age, gender, country))
        close_btn.clicked.connect(self.close)
        load_btn.clicked.connect(self.load_info)

    def save_info(self, age, gender, country):
        data = [generate_id(), age, gender, country]
        try:
            with open("./info.csv", "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(data)
            QMessageBox.critical(self, "save completed", "info save ok")
        except Exception as e:
            QMessageBox.critical(self, "not saved", f"{str(e)}")

    def load_info(self):
        try:
            with open("info.csv", "r") as f:
                reader = csv.reader(f)
                lines = [l for l in reader]
            if len(lines) > 0:
                list_win = ListWindow(lines)
                list_win.exec()
            else:
                QMessageBox.information(self, "Load info", "No such info")
        except Exception as e:
            QMessageBox.critical(self, "Failed to load info", f"{str(e)}")


class ListWindow(QDialog):
    def __init__(self, lines) -> None:
        super().__init__()
        self.setWindowTitle("Saved info")

        list_widget = QListWidget()
        for l in lines:
            item = f"ID : {l[0]}, age : {l[1]}, gender : {l[2]}, country : {l[3]}"
            list_widget.addItem(item)

        layout = QVBoxLayout()
        layout.addWidget(list_widget)
        self.setLayout(layout)


def generate_id():
    import time

    return str(int(time.time()))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    input_win = InputWindow()
    input_win.show()
    sys.exit(app.exec())
