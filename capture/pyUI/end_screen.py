from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow

class EndScreen(QMainWindow):
    def __init__(self):
        # Initialize the parent QMainWindow class
        super(EndScreen, self).__init__()

        # Load the UI elements into the StartScreen object
        uic.loadUi('./UI/stop_screen.ui', self)

        # Connect 'clicked' signal to the slot
        self.endButton.clicked.connect(self.onClickEnd)

    def onClickEnd(self):
        self.moveToNextPage()
