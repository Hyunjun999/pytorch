import sys
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QLabel,
    QWidget,
    QLineEdit,
    QMessageBox,
    QStackedWidget,
    QListWidget,
)
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker

# DB setting
engine = create_engine("sqlite:///user.db", echo=True)
Base = declarative_base()


# Schema
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True)
    password = Column(String)

    def __init__(self, name, password):
        self.username = name
        self.password = password


Session = sessionmaker(bind=engine)
session = Session()


class RegisterPage(QWidget):
    def __init__(self, stacked_widget, main_window) -> None:
        super().__init__()
        self.stacked_widget = stacked_widget
        self.main_window = main_window

        self.layout = QVBoxLayout()
        self.username_label = QLabel("UserName: ")
        self.username_input = QLineEdit()

        self.password_label = QLabel("Password: ")
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)

        self.register_btn = QPushButton("Register")
        self.register_btn.clicked.connect(self.register)

        self.layout.addWidget(self.username_label)
        self.layout.addWidget(self.username_input)
        self.layout.addWidget(self.password_label)
        self.layout.addWidget(self.password_input)
        self.layout.addWidget(self.register_btn)

        self.setLayout(self.layout)

    def register(self):
        username = self.username_input.text()
        password = self.password_input.text()

        if not username or not password:
            QMessageBox.warning(
                self, "Error", "Please enter the right username or password"
            )

        # user create
        user = User(username, password)

        # DB input
        session.add(user)
        session.commit()

        QMessageBox.information(self, "Success", "Registration was successfully done")
        self.stacked_widget.setCurrentIndex(1)
        self.main_window.show_login_page()


class LoginPage(QWidget):
    def __init__(self, stacked_widget, main_window) -> None:
        super().__init__()
        self.stacked_widget = stacked_widget
        self.main_window = main_window

        self.layout = QVBoxLayout()

        self.username_label = QLabel("UserName: ")
        self.username_input = QLineEdit()

        self.password_label = QLabel("Password: ")
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)

        self.login_btn = QPushButton("Login")
        self.login_btn.clicked.connect(self.login)

        self.register_btn = QPushButton("Register")
        self.register_btn.clicked.connect(self.register)

        self.layout.addWidget(self.username_label)
        self.layout.addWidget(self.username_input)
        self.layout.addWidget(self.password_label)
        self.layout.addWidget(self.password_input)
        self.layout.addWidget(self.login_btn)
        self.layout.addWidget(self.register_btn)

        self.setLayout(self.layout)

    def login(self):
        username = self.username_input.text()
        password = self.password_input.text()

        if not username or not password:
            QMessageBox.warning(self, "Error", "Please enter the username and password")
            return

        user = (
            session.query(User).filter_by(username=username, password=password).first()
        )

        if user:
            QMessageBox.information(self, "Success", "Login succeeded")
            self.stacked_widget.setCurrentIndex(2)

            self.main_window.show_admin_page()

        else:
            QMessageBox.warning(self, "Error", "Invalid username or password")

    def register(self):
        self.main_window.show_register_page()


class AdminPage(QWidget):
    def __init__(self, main_window) -> None:
        super().__init__()

        self.main_window = main_window
        self.layout = QVBoxLayout()
        self.user_list = QListWidget()
        self.show_user_list_btn = QPushButton("Show User List")
        self.show_user_list_btn.clicked.connect(self.show_user_list)

        self.logout_btn = QPushButton("Logout")
        self.logout_btn.clicked.connect(self.logout)

        self.layout.addWidget(self.show_user_list_btn)
        self.layout.addWidget(self.user_list)
        self.layout.addWidget(self.logout_btn)

        self.setLayout(self.layout)

    def show_user_list(self):
        self.user_list.clear()
        users = session.query(User).all()

        for u in users:
            self.user_list.addItem(u.username)

    def logout(self):
        self.main_window.show_login_page()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("User Authentication")
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.register_page = RegisterPage(self.stacked_widget, self)
        self.login_page = LoginPage(self.stacked_widget, self)
        self.admin_page = AdminPage(self)

        self.stacked_widget.addWidget(self.login_page)
        self.stacked_widget.addWidget(self.register_page)
        self.stacked_widget.addWidget(self.admin_page)

        self.show_login_page()

    def show_register_page(self):
        self.stacked_widget.setCurrentIndex(1)
        self.register_page.username_input.clear()
        self.register_page.password_input.clear()

    def show_login_page(self):
        self.stacked_widget.setCurrentIndex(0)
        self.login_page.username_input.clear()
        self.login_page.password_input.clear()

    def show_admin_page(self):
        self.stacked_widget.setCurrentIndex(2)

    def show_register_success_msg(self):
        QMessageBox.information(self, "Success", "Registration successful.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    Base.metadata.create_all(engine)
    win = MainWindow()
    win.show()

    sys.exit(app.exec())
