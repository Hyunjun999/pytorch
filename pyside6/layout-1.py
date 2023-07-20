import sys
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
)


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Complicated UI")

        # Groupbox1
        group_box1 = QGroupBox("Group box 1")
        label1 = QLabel("Name:")
        line_edit1 = QLineEdit()
        btn1 = QPushButton("Save")

        layout1 = QVBoxLayout()
        layout1.addWidget(label1)
        layout1.addWidget(line_edit1)
        layout1.addWidget(btn1)
        group_box1.setLayout(layout1)

        # Groupbox2
        group_box2 = QGroupBox("Group box 2")
        label2 = QLabel("Age:")
        line_edit2 = QLineEdit()
        btn2 = QPushButton("Cancel")

        layout2 = QHBoxLayout()
        layout2.addWidget(label2)
        layout2.addWidget(line_edit2)
        layout2.addWidget(btn2)
        group_box2.setLayout(layout2)

        # Entire layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(group_box1)
        main_layout.addWidget(group_box2)

        self.setLayout(main_layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
