from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog


def open_file():
    file_dialog = QFileDialog()
    file_dialog.setWindowTitle("Open")
    file_dialog.setFileMode(QFileDialog.ExistingFile)
    file_dialog.setViewMode(QFileDialog.Detail)
    if file_dialog.exec():
        selected = file_dialog.selectedFiles()


app = QApplication()
w = QMainWindow()
btn = QPushButton("Open", w)
btn.clicked.connect(open_file)
w.show()
app.exec()
