import sys
import csv
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QListWidget,
    QMessageBox,
)


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Complicated UI")
        self.resize(500, 250)

        # Groupbox
        group_box1 = QGroupBox("Info")
        group_box2 = QGroupBox("View Content")
        group_box3 = QGroupBox("Save and Load")

        # Label
        self.label_id = QLabel("ID:")
        self.label_age = QLabel("Age:")
        self.label_gender = QLabel("Gender:")
        self.label_country = QLabel("Country:")

        # Input line
        self.id_line_edit = QLineEdit()
        self.age_line_edit = QLineEdit()
        self.gender_line_edit = QLineEdit()
        self.country_line_edit = QLineEdit()

        # Push box
        self.view_btn = QPushButton("View")
        self.view_btn.clicked.connect(self.show_info)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close_info)

        # List box
        self.list_widget = QListWidget()

        # Save and load button
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.save_info)
        self.load_btn = QPushButton("Load")
        self.load_btn.clicked.connect(self.load_info)

        # Group box 1
        layout1 = QVBoxLayout()
        layout1.addWidget(self.label_id)
        layout1.addWidget(self.id_line_edit)
        layout1.addWidget(self.label_age)
        layout1.addWidget(self.age_line_edit)
        layout1.addWidget(self.label_gender)
        layout1.addWidget(self.gender_line_edit)
        layout1.addWidget(self.label_country)
        layout1.addWidget(self.country_line_edit)

        group_box1.setLayout(layout1)

        # Group box 2
        self.info_label = QLabel()
        layout2 = QVBoxLayout()
        layout2.addWidget(self.info_label)
        layout2.addWidget(self.view_btn)
        layout2.addWidget(self.close_btn)
        layout2.setContentsMargins(
            10,
            10,
            10,
            10,
        )
        group_box2.setLayout(layout2)

        # Group box 3
        layout3 = QVBoxLayout()
        layout3.addWidget(self.save_btn)
        layout3.addWidget(self.load_btn)
        layout3.addWidget(self.list_widget)
        layout3.setContentsMargins(10, 10, 10, 10)
        group_box3.setLayout(layout3)

        # Entire layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(group_box1)
        main_layout.addWidget(group_box2)
        main_layout.addWidget(group_box3)
        self.setLayout(main_layout)

    def show_info(self):
        id = self.id_line_edit.text()
        age = self.age_line_edit.text()
        gender = self.gender_line_edit.text()
        country = self.country_line_edit.text()

        info_text = f"ID: {id} \nAge: {age}, \nGender: {gender}, \nCountry: {country}"
        self.info_label.setText(info_text)

    def close_info(self):
        self.age_line_edit.clear()
        self.gender_line_edit.clear()
        self.country_line_edit.clear()
        self.info_label.clear()

    def save_info(self):
        data = [
            self.id_line_edit.text(),
            self.age_line_edit.text(),
            self.gender_line_edit.text(),
            self.country_line_edit.text(),
        ]
        try:
            with open("data.csv", "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(data)
            QMessageBox.information(self, "save completed", "info save ok")
        except Exception as e:
            QMessageBox.critical(self, "not saved", f"{str(e)}")

    def load_info(self):
        self.list_widget.clear()
        try:
            with open("data.csv", "r") as f:
                reader = csv.reader(f)
                for r in reader:
                    text = f"id {r[0]}, age: {r[1]}, gender: {r[2]}, country: {r[3]}"
                    self.list_widget.addItem(text)
        except Exception as e:
            QMessageBox.critical(self, "Failed to load info", f"{str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
