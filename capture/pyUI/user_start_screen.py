from PyQt5 import uic
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QMainWindow

class UserStartScreen(QMainWindow):
    def __init__(self, active_email):
        # Initialize the parent QMainWindow class
        super(UserStartScreen, self).__init__()

        # Load the UI elements into the StartScreen object
        uic.loadUi('./UI/start_screen_user.ui', self)

        # Connect 'clicked' signal to the slot
        self.startButton.clicked.connect(self.onClickSubmit)

        # Create a QFont object for the options
        font = QFont("Arial", 22)

        self.emailAddress.setFont(font)  # Set the font for the label
        # self.emailAddress.setAlignment(QtCore.Qt.AlignLeft)
        self.emailAddress.setText(active_email)

    def onClickSubmit(self):
        self.moveToNextPage()
