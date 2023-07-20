import sys
from typing import Optional
from PySide6.QtCore import Qt
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtWidgets import QApplication, QMainWindow, QTableView


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Table view example")

        table_view = QTableView(self)
        self.setCentralWidget(table_view)

        # Data model
        model = QStandardItemModel(4, 3, self)
        model.setHorizontalHeaderLabels(["Name", "Age", "Sex"])
        # Data appending
        model.setItem(0, 0, QStandardItem("Alice"))
        model.setItem(0, 1, QStandardItem("25"))
        model.setItem(0, 2, QStandardItem("F"))

        model.setItem(1, 0, QStandardItem("Bob"))
        model.setItem(1, 1, QStandardItem("30"))
        model.setItem(1, 2, QStandardItem("M"))

        model.setItem(2, 0, QStandardItem("Charlie"))
        model.setItem(2, 1, QStandardItem("35"))
        model.setItem(2, 2, QStandardItem("M"))

        model.setItem(3, 0, QStandardItem("Daisy"))
        model.setItem(3, 1, QStandardItem("28"))
        model.setItem(3, 2, QStandardItem("F"))

        table_view.setModel(model)
        table_view.resizeColumnsToContents()
        table_view.setEditTriggers(QTableView.NoEditTriggers)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
