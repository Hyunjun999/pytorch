import sys
import os
from typing import Optional
import PySide6.QtCore
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QFileDialog,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
)


class FileExplorer(QMainWindow):
    def __init__(self):
        super(FileExplorer, self).__init__()

        self.setWindowTitle("Explorer")
        self.resize(500, 400)
        self.folder_btn = QPushButton("Open Folder")
        self.folder_btn.clicked.connect(self.open_folder_dialog)

        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["File"])

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.folder_btn)
        main_layout.addWidget(self.tree_widget)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.path = ""

    def open_folder_dialog(self):
        folder_dialog = QFileDialog(self)
        folder_dialog.setFileMode(QFileDialog.Directory)
        folder_dialog.setOption(QFileDialog.ShowDirsOnly, True)
        folder_dialog.directoryEntered.connect(self.set_folder_path)
        folder_dialog.accepted.connect(self.display_files)
        folder_dialog.exec()

    def set_folder_path(self, path):
        self.path = path

    def display_files(self):
        if self.path:
            self.tree_widget.clear()

            root = QTreeWidgetItem(self.tree_widget, [self.path])
            self.tree_widget.addTopLevelItem(root)

            for dir_path, _, file_names in os.walk(self.path):
                dir_item = QTreeWidgetItem(root, [os.path.basename(dir_path)])
                root.addChild(dir_item)

                for f in file_names:
                    file_item = QTreeWidgetItem(dir_item, [f])
                    dir_item.addChild(file_item)

                root.setExpanded(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = FileExplorer()
    w.show()
    sys.exit(app.exec())
